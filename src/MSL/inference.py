import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

arm = XArmAPI('192.168.1.219')
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)
arm.motion_enable(enable=True)
arm.set_state(0)
arm.set_mode(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Connect to cameras
ctx = rs.context()
devices = ctx.query_devices()

#if len(devices) < 2:
    #raise RuntimeError("Need at least two RealSense cameras connected")

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
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    pipelines.append(pipeline)
    configs.append(config)

#Create action chunk
dt = 0.2 # your timestep, may have to tweak for finer motor control
while True:
    frames_wrist = pipelines[0].wait_for_frames()
    frames_exterior = pipelines[1].wait_for_frames()

    wrist = frames_wrist.get_color_frame()
    exterior = frames_exterior.get_color_frame()

    wrist = np.asanyarray(wrist.get_data())
    exterior = np.asanyarray(exterior.get_data())

    wrist = np.flip(wrist, axis=2)
    exterior = np.flip(exterior, axis=2)

    a = cv2.resize(wrist, (224, 224))
    b = cv2.resize(exterior, (224, 224))

    code, angles = arm.get_servo_angle(is_radian=True)
    code, g_p = arm.get_gripper_position()
    state = np.array(angles)
    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image": b,
        "observation/wrist_image": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state[:6],
        "prompt": "point up",
    }

    # Run inference 
    action_chunk = np.array(policy.infer(observation)["actions"])
    for i in range(0, 50):
        cmd_joint_pose = state[:6] + action_chunk[i,:6]
        cmd_gripper_pose = (g_p + action_chunk[i,6]) * -860 + 850 # denormalize the gripper action
        print(cmd_joint_pose)
        arm.set_servo_angle(servo_id=8, angle=cmd_joint_pose, is_radian=True) 
        arm.set_gripper_position(cmd_gripper_pose)
        time.sleep(0.01)
 

