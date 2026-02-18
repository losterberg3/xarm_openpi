import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

FPS = 20.0
DT = 1.0 / FPS # your timestep
CONTROL_HZ = 40.0 # keep as a multiple of 10
ACTION_ROLLOUT = 30

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
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_xarm_finetune/lars_history_exp_v1/25000")

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
    pose[3] = pose[3] % 360
    pose[5] = pose[5] % 360
    angles_rad = (np.array(pose[3:6]) * np.pi / 180).tolist()
    state = np.array(pose[:3] + angles_rad, dtype=np.float32)
    code, g_p = arm.get_gripper_position()
    g_p = np.array((g_p - 850) / -860)

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state,
        "prompt": "Drop the block in the cup and then knock that same cup over",
    }
    return observation

def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ)

    for i in range(int(DT * CONTROL_HZ)):
        start = time.perf_counter()
        command = state + delta_increment
        command[3] = (command[3]+ 180) % 360 -180
        command[5] = (command[5]+ 180) % 360 -180

        x, y, z, roll, pitch, yaw = command
        print(x, y, z, roll, pitch, yaw)

        arm.set_servo_cartesian(command, speed=100, mvacc=1000)

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
    print(init_joints)
    
    while count < ACTION_ROLLOUT:
        # grab current state
        t0 = time.perf_counter()
        pose = arm.get_position()[1]
        pose[3] = pose[3] % 360
        pose[5] = pose[5] % 360
        state = np.array(pose, dtype=np.float32)
        
        # get the target angles
        cmd_joint_pose = np.array(action[count,:6])
        cmd_joint_pose[3:6] = cmd_joint_pose[3:6] / np.pi * 180
        
        # execute smooth motion to target via interpolation
        interpolate_action(state, cmd_joint_pose)
        
        cmd_gripper_pose = (action[count,6]) * -860 + 850 # unnormalize the gripper action
        arm.set_gripper_position(cmd_gripper_pose)

        count += 1
        time_left = DT - (time.perf_counter() - t0)
        
        time.sleep(max(time_left,0))
    