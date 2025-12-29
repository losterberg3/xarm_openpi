import time
import numpy as np
import pyrealsense2 as rs
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from xarm.wrapper import XArmAPI
from pathlib import Path

# ------------------------
# Config
# ------------------------
REPO_NAME = "lars/xarm_demos"
FPS = 10
DT = 1.0 / FPS
ARM_IP = "192.168.1.219"

todo
TASK_DESCRIPTION = "Rotate the gripper"

START_FLAG = Path("/tmp/start_demo")
STOP_FLAG  = Path("/tmp/stop_demo")

# ------------------------
# Init robot + cameras
# ------------------------
arm = XArmAPI(ARM_IP)
arm.connect()

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
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
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
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": { # this one is not used, put it as zeros or something
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # We will use joint *velocity* actions here (6D) + gripper position (1D)
                "names": ["actions"],
            },
        },
    )
    print("Dataset created, waiting for start signal.")

# ------------------------
# Collect episode
# ------------------------

recording = False
prev = None

try:
    while True:
        if START_FLAG.exists() and not recording:
            START_FLAG.unlink()
            print("Starting demo")

            prev_joints = arm.get_servo_angle(is_radian=True)[1][:6]
            prev_gripper = (arm.get_gripper_position()[1] - 850) / -860
            prev = np.array(prev_joints + [prev_gripper], dtype=np.float32)
            recording = True

        if STOP_FLAG.exists() and recording:
            STOP_FLAG.unlink()
            print("Ending demo")
            prev = None
            recording = False
            dataset.save_episode()
            print("Episode saved")

        if not recording:
            time.sleep(0.05)
            continue

        start = time.perf_counter()

        # ---- Observation ----
        joints = arm.get_servo_angle(is_radian=True)[1][:6]  # returns tuple (ret_code, data)
        # Read gripper position (if you have a Robotiq gripper, might be different API)
        gripper = (arm.get_gripper_position()[1] - 850) / -860

        curr = np.array(joints + [gripper], dtype=np.float32)
        action = curr - prev
        prev = curr
        
        wrist, base, base2 = read_cameras()

        # ---- Write frame ----
        dataset.add_frame(
            {
                "joint_position": np.asarray(joints, dtype=np.float32),
                "gripper_position": np.asarray([gripper], dtype=np.float32),
                "actions": action,
                "exterior_image_1_left": base,
                "exterior_image_2_left": base2,
                "wrist_image_left": wrist,
                "task": TASK_DESCRIPTION,
            }
        )

        # ---- Timing ----
        elapsed = time.perf_counter() - start
        time.sleep(max(0.0, DT - elapsed))

except KeyboardInterrupt:
    print("Shutting down")

finally:
    for p in pipelines:
        p.stop()
    arm.disconnect()