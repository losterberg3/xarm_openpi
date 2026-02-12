import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

DT = 0.25 # your timestep
CONTROL_HZ = 40.0 # keep as a multiple of 10
ACTION_ROLLOUT = 20

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
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_xarm_finetune/lars_abs_pos_2_6/20000")

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
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
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

    code, angles = arm.get_servo_angle(is_radian=True)
    code, g_p = arm.get_gripper_position()
    state = np.array(angles)
    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state[:6],
        "prompt": "Grab the yellow bottle and place it on the pink marker",
    }
    return observation

def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ)

    for i in range(int(DT * CONTROL_HZ)):
        start = time.perf_counter()
        state = state + delta_increment
        arm.set_servo_angle_j(state, is_radian=True, wait=True)
        time_left = (1 / CONTROL_HZ) - (time.perf_counter() - start)
        time.sleep(max(time_left,0))
       
while True:

    observation = get_observation()

    print("Running inference")
    inference = policy.infer(observation)
    
    action = np.array(inference["actions"])
    
    count = 0

    init_joints = observation["observation/joint_position"]
    init_gripper = observation["observation/gripper_position"]
    print("state")
    print(init_gripper)
    DT = 3.0
    while count < ACTION_ROLLOUT:
        t0 = time.perf_counter()
        code, angles = arm.get_servo_angle(is_radian=True)
        state = np.array(angles)
        
        # get the target angles
        cmd_joint_pose = np.array(action[count,:6])
        
        # execute smooth motion to target via interpolation
        interpolate_action(state[:6], cmd_joint_pose)
        print("command")
        #print(cmd_joint_pose)
        cmd_gripper_pose = (action[count,6]) * -860 + 850 # unnormalize the gripper action
        print(cmd_gripper_pose)
        #arm.set_servo_angle(servo_id=8, angle=cmd_joint_pose, is_radian=True) 
        arm.set_gripper_position(cmd_gripper_pose)

        count += 1
        time_left = DT - (time.perf_counter() - t0)
        DT = 0.125
        time.sleep(max(time_left,0))
    