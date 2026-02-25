import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time

from scipy.spatial.transform import Rotation

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

FPS = 3.0
DT = 1.0 / FPS # your timestep
CONTROL_HZ = 40.0 # keep as a multiple of 10
ACTION_ROLLOUT = 10

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
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_xarm_finetune/lars_history_exp_v2/25000")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir, language_out=False)

# Connect to cameras
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    raise RuntimeError("Need at least two RealSense cameras connected")

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

def get_observation():
    frames_wrist = pipelines[0].wait_for_frames()
    frames_exterior = pipelines[1].wait_for_frames()

    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()

    a = np.asanyarray(wrist.get_data())
    b = np.asanyarray(exterior.get_data())

    pose = arm.get_position()[1]
    rot = Rotation.from_euler('xyz', pose[3:6], degrees=True)
    quat = rot.as_quat()

    state = np.concatenate([pose[:3], quat]).astype(np.float32)

    code, g_p = arm.get_gripper_position()
    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/eef_position": state,
        "prompt": "Drop the block in the box and then tap that box",
    }
    return observation

while True:
    observation = get_observation()
    print("Running inference")
    inference = policy.infer(observation)
    action = np.array(inference["actions"])
    
    # 1. Get current physical state for smoothing and flip-protection
    _, current_pose = arm.get_position()
    current_pos = np.array(current_pose[:3])
    current_euler = np.array(current_pose[3:6])
    # Convert current orientation to quat to use as a "reference"
    ref_quat = Rotation.from_euler('xyz', current_euler, degrees=True).as_quat()

    for count in range(ACTION_ROLLOUT):
        t0 = time.perf_counter()
        
        target_xyz = action[count, :3]
        raw_quat = action[count, 3:7]

        # 2. Normalize and Fix Double Cover (The "Wrong Direction" fix)
        target_quat = raw_quat / np.linalg.norm(raw_quat)
        if np.dot(target_quat, ref_quat) < 0:
            target_quat = -target_quat # Flip quat to stay in the same hemisphere
        
        # 3. Convert to Euler
        target_euler = Rotation.from_quat(target_quat).as_euler('xyz', degrees=True)

        # 4. Clamp Pitch (Singularity protection)
        if abs(target_euler[1]) > 88.0:
            target_euler[1] = np.sign(target_euler[1]) * 88.0

        # 5. YOUR SMOOTHING (Now properly updating the command)
        if count == 0:
            dist = np.linalg.norm(target_xyz - current_pos)
            if dist > 20: 
                print(f"Warning: Large jump detected ({dist:.2f}mm). Smoothing...")
                target_xyz = current_pos + (target_xyz - current_pos) * 0.2

        # 6. RECONSTRUCT CMD_POSE (Crucial step you caught earlier)
        cmd_pose = np.concatenate([target_xyz, target_euler])

        # 7. Execute
        print(f"Executing: {cmd_pose}")
        arm.set_servo_cartesian(cmd_pose, speed=20, mvacc=500)

        # Update ref_quat so the next step in the chunk stays consistent
        ref_quat = target_quat

        # Gripper & Timing
        cmd_gripper_pose = (action[count, 7]) * -860 + 850 
        arm.set_gripper_position(cmd_gripper_pose, wait=False)

        time_left = DT - (time.perf_counter() - t0)
        time.sleep(max(time_left, 0))