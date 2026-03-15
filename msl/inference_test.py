import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

config = _config.get_config("pi05_xarm_gru")
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_gru_addition/gru_exp_v1/14999")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

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

    state = np.zeros(6)
    g_p = np.zeros(1)   
    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": g_p,
        "observation/joint_position": state,
        "prompt": "Drop the block in the cup and then knock that same cup over",
    }
    return observation

history = None
while True:
    observation = get_observation()
    inference = policy.infer(observation, history)
    
    action = np.array(inference["actions"])
    history_init = np.array(inference["history"])

    init_joints = observation["observation/joint_position"]
    init_gripper = observation["observation/gripper_position"]
    if history is not None:
        diff = history_init - history
        # GRU update diagnostics: L2 norm of change, max abs diff, mean abs history (scale)
        update_l2 = float(np.linalg.norm(diff))
        update_max = float(np.max(np.abs(diff)))
        hist_scale = float(np.mean(np.abs(history_init)))
        print(f"GRU update: L2={update_l2:.2f} max|Δ|={update_max:.4f}  (history mean|·|={hist_scale:.4f})")
    history = history_init