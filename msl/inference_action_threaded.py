import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time
import threading

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

FPS = 20.0
DT = 1.0 / FPS 
CONTROL_HZ = 40.0     
ACTION_ROLLOUT = 20      
CAMERA_FRAMES_TIMEOUT = 0.5

arm = XArmAPI('192.168.1.219')
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_xarm_finetune/lars_eef_2_11/25000")

policy = policy_config.create_trained_policy(config, checkpoint_dir, language_out=False)

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 2:
    raise RuntimeError("Need at least two RealSense cameras connected")

serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)

pipelines = []
configs = []
for serial in serials:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 320, 240, rs.format.rgb8, 30)
    pipeline.start(cfg)
    pipelines.append(pipeline)
    configs.append(cfg)

latest_wrist_frame = None
latest_exterior_frame = None
frames_lock = threading.Lock()
stop_event = threading.Event()
action_lock = threading.Lock()

latest_actions = None
new_actions_event = threading.Event()

def get_arm_state():
    pose = arm.get_position()[1]
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3] + angles_rad, dtype=np.float32)
    code, g_p = arm.get_gripper_position()
    g_p = np.array((g_p - 850) / -860, dtype=np.float32)
    return state, g_p


def camera_grabber():
    global latest_wrist_frame, latest_exterior_frame
    # pipelines[0] => wrist, pipelines[1] => exterior
    while not stop_event.is_set():
        try:
            # wrist
            frames_wrist = pipelines[0].wait_for_frames(timeout_ms=int(CAMERA_FRAMES_TIMEOUT*1000))
            frames_exterior = pipelines[1].wait_for_frames(timeout_ms=int(CAMERA_FRAMES_TIMEOUT*1000))

            wrist = frames_wrist.get_color_frame()
            exterior = frames_exterior.get_color_frame()
            if not wrist or not exterior:
                continue

            a = np.asanyarray(wrist.get_data())
            b = np.asanyarray(exterior.get_data())

            with frames_lock:
                latest_wrist_frame = a.copy()
                latest_exterior_frame = b.copy()
        except Exception as e:
            time.sleep(0.001)
            continue

def build_observation_from_latest():
    with frames_lock:
        wrist = None if latest_wrist_frame is None else latest_wrist_frame.copy()
        exterior = None if latest_exterior_frame is None else latest_exterior_frame.copy()

    if wrist is None or exterior is None:
        return None

    state, g_p = get_arm_state()
    observation = {
        "observation/exterior_image_1_left": exterior,
        "observation/wrist_image_left": wrist,
        "observation/gripper_position": g_p,
        "observation/joint_position": state,
        "prompt": "Grab the yellow bottle and place it on the pink marker",
    }
    return observation

def inference_worker():
    global latest_actions
    while not stop_event.is_set():
        obs = build_observation_from_latest()
        if obs is None:
            time.sleep(0.005)
            continue

        inference = policy.infer(obs)
        actions = np.array(inference["actions"])

        with action_lock:
            latest_actions = actions.copy()
            new_actions_event.set()

def control_loop():
    action_index = 0
    new_actions = None
    old_actions = None
    blend = False
    b = 5
    blend_i = 0

    while new_actions is None:
        if new_actions_event.is_set():
            with action_lock:
                new_actions = latest_actions.copy()
                new_actions_event.clear()

    while not stop_event.is_set():
        t0 = time.perf_counter()

        if blend:
            alpha = blend_i / float(b) 
            cmd_pose = alpha * new_actions[0,:] + (1 - alpha) * old_actions[old_index,:]
            old_index += 1  
            blend_i += 1  
            if blend_i >= b:
                blend = False
                action_index = 1        
        else:
            cmd_pose = new_actions[action_index,:]
            action_index += 1

        cmd_pose[3:6] = cmd_pose[3:6] / np.pi * 180
        cmd_pose[3] = (cmd_pose[3] + 180) % 360 - 180
        cmd_pose[5] = (cmd_pose[5] + 180) % 360 - 180
        
        print(cmd_pose)
        arm.set_servo_cartesian(cmd_pose[:6], speed=80, mvacc=500)

        cmd_gripper_pose = cmd_pose[6] * -860 + 850
        arm.set_gripper_position(cmd_gripper_pose)

        if new_actions_event.is_set():
            with action_lock:
                if action_index > 15:
                    blend = True
                    blend_i = 0 
                    old_actions = new_actions
                    old_index = action_index
                    new_actions = latest_actions.copy()
                new_actions_event.clear()

        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left,0))

camera_thread = threading.Thread(target=camera_grabber, daemon=True)
inference_thread = threading.Thread(target=inference_worker, daemon=True)

camera_thread.start()
inference_thread.start()

try:
    control_loop()
except KeyboardInterrupt:
    print("Stopping by user request...")
finally:
    stop_event.set()
    # allow threads to exit briefly
    time.sleep(0.2)
    # optionally stop pipelines cleanly
    for p in pipelines:
        try:
            p.stop()
        except Exception:
            pass
    print("Clean shutdown complete.")
