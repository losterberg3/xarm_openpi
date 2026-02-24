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

FPS = 20.0
DT = 1.0 / FPS # your timestep
CONTROL_HZ = 40.0 # keep as a multiple of 10
ACTION_ROLLOUT = 30
"""
arm = XArmAPI('192.168.1.219')
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)
arm.motion_enable(enable=True)
arm.set_mode(1)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)
"""

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_xarm_finetune/lars_history_exp_v2/25000")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir, language_out=False)
"""
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
"""
def get_observation():
    #frames_wrist = pipelines[0].wait_for_frames()
    #frames_exterior = pipelines[1].wait_for_frames()

    #wrist = frames_wrist.get_color_frame()
    #exterior = frames_exterior.get_color_frame()

    #a = np.asanyarray(wrist.get_data())
    #b = np.asanyarray(exterior.get_data())

    a = np.zeros((3,320,240))
    b = a

    #pose = arm.get_position()[1]
    #rot = Rotation.from_euler('xyz', pose[3:6], degrees=True)
    #quat = rot.as_quat()

    #state = np.concatenate([pose[:3], quat]).astype(np.float32)

    #code, g_p = arm.get_gripper_position()
    #g_p = np.array((g_p - 850) / -860)

    state = np.array([283.1846618652344,
        -12.364502906799316,
        435.26678466796875,
        -0.195, 
        0.810, 
        0.058, 
        0.551])

    g_p = np.array([0.4209335446357727])

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

    
    init_joints = observation["observation/eef_position"]
    init_gripper = observation["observation/gripper_position"]
    print("initial eef")
    print(Rotation.from_quat(init_joints[3:7]).as_euler('xyz', degrees=True))
    
    for count in range(ACTION_ROLLOUT):
        t0 = time.perf_counter()
        
        target_xyz = action[count, :3]
        raw_quat = action[count, 3:7]

        norm = np.linalg.norm(raw_quat)
        clean_quat = raw_quat / norm
        
        rot_obj = Rotation.from_quat(clean_quat)
        target_euler = rot_obj.as_euler('xyz', degrees=True)

        cmd_pose = np.concatenate([target_xyz, target_euler])

        print(cmd_pose)
        
        cmd_gripper_pose = (action[count,7]) * -860 + 850 # unnormalize the gripper action
        #arm.set_gripper_position(cmd_gripper_pose)

        count += 1
        time_left = DT - (time.perf_counter() - t0)
        
        time.sleep(max(time_left,0))
