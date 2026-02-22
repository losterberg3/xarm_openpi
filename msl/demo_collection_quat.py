import time
import sys
import select
import threading
import numpy as np
import pyrealsense2 as rs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from xarm.wrapper import XArmAPI
from pathlib import Path
from scipy.spatial.transform import Rotation

# ------------------------
# Config
# ------------------------
REPO_NAME = "lars/xarm_history_exp_v2"
FPS = 20.0
DT = 1.0 / FPS
ARM_IP = "192.168.1.219"

TASK_DESCRIPTION = "Drop the block in the box and then tap that box"

START_FLAG = Path("/tmp/start_demo")
STOP_FLAG  = Path("/tmp/stop_demo")

if START_FLAG.exists():
    START_FLAG.unlink()

if STOP_FLAG.exists():
    STOP_FLAG.unlink()


# ------------------------
# Init robot + cameras
# ------------------------
arm = XArmAPI(ARM_IP)
arm.connect()


_robot_lock = threading.Lock()
_latest_pose = None        # will store the full pose list from arm.get_position()[1]
_latest_gripper = None     # will store the scalar gripper raw value

_poller_running = True     # used to stop the poller on shutdown

def _robot_poller():
    """Continuously poll the robot for state and update the cached variables.

    Keep this as tight as reasonable but yield occasionally to avoid hammering.
    """
    global _latest_pose, _latest_gripper, _poller_running
    while _poller_running:
        try:
            # These are the blocking calls we want to isolate from the main loop
            pos_result = arm.get_position()
            gripper_result = arm.get_gripper_position()

            if pos_result is not None and len(pos_result) > 1:
                pose = pos_result[1]
            else:
                pose = None

            if gripper_result is not None and len(gripper_result) > 1:
                gr = gripper_result[1]
            else:
                gr = None

            # write into cached vars under lock
            with _robot_lock:
                if pose is not None:
                    _latest_pose = pose[:]   # make a shallow copy (list)
                if gr is not None:
                    _latest_gripper = gr

        except Exception as e:
            # don't crash poller on intermittent errors; print once in a while
            # (you can replace with logging)
            print("robot poller error:", e)
            time.sleep(0.01)

        # Sleep a small amount; controller runs ~8 ms so polling at ~200 Hz is pointless.
        # 50 Hz is plenty to keep the cache fresh; reduce CPU usage.
        time.sleep(0.01)  # 10 ms


# Start poller thread BEFORE entering main loop
_poller_thread = threading.Thread(target=_robot_poller, daemon=True)
_poller_thread.start()

# Connect to cameras
ctx = rs.context()
devices = ctx.query_devices()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)
# check serials for which camera is which, second one is currently the external viewer
pipelines = []
configs = []

# Enable streams
for serial in serials:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    # Enable streams (color + depth if you want)
    config.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
    pipeline.start(config)
    pipelines.append(pipeline)
    configs.append(config)

def read_cameras():
    frames_wrist = pipelines[0].wait_for_frames()
    frames_exterior = pipelines[1].wait_for_frames()
    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()
    wrist = np.asanyarray(wrist.get_data())
    exterior = np.asanyarray(exterior.get_data())
    exterior2 = np.zeros_like(exterior)
    return wrist, exterior, exterior2

def timed_input(prompt, timeout, default="y"):
    print(prompt, end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip().lower()
    else:
        print(f"\nNo response after {timeout}s → defaulting to '{default}'")
        return default

# ------------------------
# Create dataset
# ------------------------

dataset_path = HF_LEROBOT_HOME / REPO_NAME

if dataset_path.exists(): 
    dataset = LeRobotDataset(
        root=dataset_path,
        repo_id=REPO_NAME,
    )
    print("Adding to existing dataset, waiting for signal.")
else:
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm",
        fps=FPS,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": { # this one is not used, put it as zeros or something
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "eef_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["eef_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                "names": ["actions"],
            },
        },
    )
    print("Dataset created, waiting for start signal.")

# ------------------------
# Collect episode
# ------------------------

recording = False
prev_data = None  # Buffer to hold the observation for time t

try:
    while True:
        if START_FLAG.exists() and not recording:
            START_FLAG.unlink()
            print("Starting demo")
            recording = True
            prev_data = None # Reset buffer for new demo

        if STOP_FLAG.exists() and recording:
            STOP_FLAG.unlink()
            print("Ending demo")
            
            recording = False
            prev_data = None # Clear buffer
            resp = timed_input("Save this demo? [y/n]: ", timeout=6, default="y")

            if resp == "y":
                dataset.save_episode()
                print("Episode saved")
            else:
                dataset = LeRobotDataset(root=dataset_path, repo_id=REPO_NAME)
                print("Episode discarded")

        if not recording:
            time.sleep(0.05)
            continue

        start = time.perf_counter()
        # 1. Capture CURRENT state (Time t+1 relative to prev_data)
        #joints = arm.get_servo_angle(is_radian=True)[1][:6]
        with _robot_lock:
            cached_pose = None if _latest_pose is None else list(_latest_pose)
            cached_gr = _latest_gripper

        if cached_pose is None or cached_gr is None:
            # Cache not warmed yet; wait a tiny bit and continue so we don't block.
            # You can change behavior to busy-wait a few ms if you prefer.
            time.sleep(0.005)
            continue

        pose = cached_pose
        rot = Rotation.from_euler('xyz', pose[3:6], degrees=True)
        quat = rot.as_quat() 
    
        gripper = (cached_gr - 850) / -860
        curr_state = np.concatenate((
            np.asarray(pose[:3], dtype=np.float32),
            quat.astype(np.float32),
            np.array([gripper], dtype=np.float32)
        ))
        
        wrist, base, base2 = read_cameras()
    
        # 2. If we have a previous observation, record it with CURRENT state as the action
        if prev_data is not None:
            dataset.add_frame(
                {
                    "eef_position": prev_data["eef"],
                    "gripper_position": prev_data["gripper"],
                    "actions": curr_state,  # This is the "future" state reached
                    "exterior_image_1_left": prev_data["base"],
                    "exterior_image_2_left": prev_data["base2"],
                    "wrist_image_left": prev_data["wrist"],
                    "task": TASK_DESCRIPTION,
                }
            )

        # 3. Store current observations to be paired with the next frame's state
        prev_data = {
            "eef": curr_state[:7],
            "gripper": curr_state[-1:],
            "wrist": wrist,
            "base": base,
            "base2": base2
        }

        # ---- Timing ----
        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, DT - elapsed))

except KeyboardInterrupt:
    print("Shutting down")

finally:
    _poller_running = False
    _poller_thread.join(timeout=1.0)

    for p in pipelines:
        p.stop()
    arm.disconnect()