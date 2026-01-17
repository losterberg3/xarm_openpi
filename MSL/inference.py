import numpy as np
import pyrealsense2 as rs
#from xarm.wrapper import XArmAPI
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer

"""
arm = XArmAPI('192.168.1.219')
if arm.get_state() != 0:
    arm.clean_error()
    time.sleep(0.5)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)
arm.set_gripper_enable(enable=True)
arm.set_gripper_mode(0)
"""

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

    #a = np.zeros((3, 640, 480), dtype=float)
    #b = a

    state = np.array([-0.20991884171962738,
        0.2138545662164688,
        -0.9285001158714294,
        -0.39744529128074646,
        -0.06720831245183945,
        3.5203089714050293])

    g_p = np.array([0.22542054951190948])
    #code, angles = arm.get_servo_angle(is_radian=True)
    #code, g_p = arm.get_gripper_position()
    #state = np.array(angles)
    #g_p = np.array((g_p - 850) / -860)

    prompt = input("Enter prompt for this observation: ").strip()

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state[:6],
        "prompt": prompt,
    }
    return observation

#Create action chunk
dt = 0.1 # your timestep, may have to tweak for finer motor control
while True:
    try:
        observation = get_observation()

        print("Running inference")
        inference = policy.infer(observation)

    except KeyboardInterrupt:
        print("\nInference interrupted, continuing loop...")
        continue
    #action = np.array(inference["actions"])
    #text_tokens = inference["text_tokens"]

    #tokenizer = PaligemmaTokenizer(max_len=200)  # Or whatever max_len you're using
    
    #tokens_list = text_tokens #.tolist()  # Get first batch element as Python list
    #decoded_text = tokenizer._tokenizer.decode(tokens_list)

    #print(f"Generated text: {decoded_text}")
    """
    count = 0
    while count < 20:
        t0 = time.perf_counter()

        state = np.array([-0.20991884171962738,
            0.2138545662164688,
            -0.9285001158714294,
            -0.39744529128074646,
            -0.06720831245183945,
            3.5203089714050293])

        g_p = np.array([0.22542054951190948])
        #code, angles = arm.get_servo_angle(is_radian=True)
        #code, g_p = arm.get_gripper_position()
        #state = np.array(angles)
        #g_p = np.array((g_p - 850) / -860)

        delta = np.array(action[count,:6])
        cmd_joint_pose = state[:6] + delta

        cmd_gripper_pose = (g_p + action[count,6]) * -860 + 850 # denormalize the gripper action
        #print(cmd_joint_pose)
        #arm.set_servo_angle(servo_id=8, angle=cmd_joint_pose, is_radian=True) 
        #arm.set_gripper_position(cmd_gripper_pose)

        count += 1
        time_left = dt - (time.perf_counter() - t0)
        time.sleep(max(time_left,0))
    """
 

