import numpy as np
import pyrealsense2 as rs
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.models.tokenizer import PaligemmaTokenizer


base = True
jit = False
output_len = 200

if base:
    config = _config.get_config("pi05_base")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
else:
    config = _config.get_config("pi05_xarm")
    checkpoint_dir = download.maybe_download("/home/larsosterberg/MSL/openpi/checkpoints/pi05_xarm_finetune/lars_abs_pos/24999")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir, language_out=True, jitted_language=jit)
# make sure to edit tokenizer.py if you want language to only include the prompt

tokenizer = PaligemmaTokenizer

# Connect to cameras
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    raise RuntimeError("Need at least two RealSense cameras connected")

serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)
# Found cameras: ['317222072257', '244222071219'], if this ever changes, then we have a problem
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

    # use norm stats for state, this doesn't end up going anywhere
    # since we omit it from the prompt
    state = np.array([-0.20991884171962738,
        0.2138545662164688,
        -0.9285001158714294,
        -0.39744529128074646,
        -0.06720831245183945,
        3.5203089714050293])

    g_p = np.array([0.22542054951190948])

    prompt = input("Enter prompt for this observation: ").strip()

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state[:6],
        "prompt": prompt,
        "history": "I just grabbed that motherfucker and put it on the motherfucking stove, ya feel me dawg. It was a successful grasp."
    }
    return observation

# query the policy
if jit:
    while True:
        try:
            observation = get_observation()

            print("Running inference")
            inference = policy.infer(observation)
            
            token_list = inference["text_tokens"].tolist()
            for item in token_list:
                decoded_text = tokenizer._tokenizer.decode(item)
                print(f"{decoded_text}", end=" ", flush=True)
            

        except KeyboardInterrupt:
            print("\nInference interrupted, continuing loop...")
            continue         
else:        
    while True:
        try:
            observation = get_observation()
            print("Running inference")
            
            # Generate all tokens at once
            inference = policy.infer(observation)
            
        except KeyboardInterrupt:
            print("\nInference interrupted, continuing loop...")
            continue


"""
while True:
        try:
            observation = get_observation()

            print("Running inference")
            for i in range(output_len):
                inference = policy.infer(observation)
                token_list = inference["text_tokens"].tolist()
                if token_list[0] == 1:
                    decoded_text = " And,"
                else:
                    decoded_text = tokenizer._tokenizer.decode(token_list)
                
                print(f"{decoded_text}", end="", flush=True)
                observation["prompt"] = observation["prompt"] + decoded_text

        except KeyboardInterrupt:
            print("\nInference interrupted, continuing loop...")
            continue
"""