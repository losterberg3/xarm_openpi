#!/usr/bin/env python3

# Adapted from https://github.com/IliaLarchenko/lerobot_random/blob/main/vla/pi/evaluate_pi0.py

import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
import time
import numpy as np

# Import Pi0 model from openpi
from openpi.training import config as pi0_config
from openpi.policies import policy_config


# Configuration
FPS = 10
TASK_DESCRIPTION = "grab the yellow bottle and place it on the pink marker"
ACTIONS_TO_EXECUTE = 20  # Execute this many actions from each predicted chunk


arm = XArmAPI('192.168.1.219')
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)
arm.motion_enable(enable=True)
arm.set_state(0)
arm.set_mode(1)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("/home/larsosterberg/MSL/openpi/checkpoints/pi05_xarm_finetune/lars_test/2999")

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

print(f"Robot connected. Starting control loop with task: '{TASK_DESCRIPTION}'")
print(f"Will execute {ACTIONS_TO_EXECUTE} actions from each predicted chunk")

def get_observation():
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
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state[:6],
        "prompt": TASK_DESCRIPTION,
    }
    return observation

# Control loop variables
step = 0
last_actions = None
action_index = 0
pred_times = []

while True:
    t0 = time.perf_counter()
    
    # Run prediction when we need new actions (either first time or when we've executed enough actions)
    if last_actions is None or action_index >= min(ACTIONS_TO_EXECUTE, len(last_actions)):
        # Get robot observation
        observation = get_observation()
        
        # Run inference
        t_pred_start = time.perf_counter()
        output = policy.infer(observation)
        t_pred = time.perf_counter() - t_pred_start
        
        # Keep track of last 10 prediction times
        pred_times.append(t_pred)
        pred_times = pred_times[-10:]  # Keep last 10
        
        last_actions = output["actions"]
        action_index = 0

        print(f"Step {step}: Predicted {len(last_actions)} actions, will execute {min(ACTIONS_TO_EXECUTE, len(last_actions))}")
        print(f"Prediction took {t_pred:.3f}s (avg over last {len(pred_times)}: {np.mean(pred_times):.3f}s)")
    
    # Execute action
    else:
        # Get current action from the sequence
        action = np.array(last_actions[action_index])
        
        code, g_p = arm.get_gripper_position()

        g_p = np.array((g_p - 850) / -860)

        cmd_joint_delta = action[:6]
        cmd_gripper_pose = (g_p + action[6]) * -860 + 850
        print(cmd_joint_delta)
        arm.set_servo_angle(servo_id=8, angle=cmd_joint_delta, relative=True, is_radian=True, wait=False) 
        arm.set_gripper_position(cmd_gripper_pose)
        
        action_index += 1
        step += 1
    
    elapsed = time.perf_counter() - t0
    sleep_time = 1.0 / FPS - elapsed
    if sleep_time > 0:
        # Precise busy wait for robotics
        end_wait = time.perf_counter() + sleep_time
        while time.perf_counter() < end_wait:
            pass