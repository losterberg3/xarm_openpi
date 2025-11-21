import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import cv2
import time

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

arm = XArmAPI('192.168.1.219')
arm.motion_enable(enable=True)
arm.set_state(0)
arm.set_mode(0)
#arm.set_gripper_enable(enable=True)
#arm.set_gripper_mode(0)

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Connect to cameras
ctx = rs.context()
devices = ctx.query_devices()

#if len(devices) < 2:
    #raise RuntimeError("Need at least two RealSense cameras connected")

serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)

pipelines = []
configs = []

# Enable streams
for serial in serials:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    # Enable streams (color + depth if you want)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    pipelines.append(pipeline)
    configs.append(config)

#Create action chunk
while True:
    frames_wrist = pipelines[0].wait_for_frames()
    #frames_exterior = pipelines[1].wait_for_frames()

    wrist = frames_wrist.get_color_frame()
    #exterior = frames_exterior.get_color_frame()

    wrist = np.asanyarray(wrist.get_data())
    #exterior = np.asanyarray(exterior.get_data())

    wrist = np.flip(wrist, axis=2)
    #exterior = np.flip(exterior, axis=2)

    a = cv2.resize(wrist, (224, 224))
    #b = cv2.resize(exterior, (224, 224))

    code, angles = arm.get_servo_angle(is_radian=True)
    #code, g_p = arm.get_gripper_position()
    state = np.array(angles)
    #g_p = np.array(g_p)
    print(state)
    #print(g_p)
    observation = {
        "observation/exterior_image": a,
        "observation/wrist_image": a,
        "observation/gripper_position": state[6],
        "observation/joint_position": state[:6],
        "prompt": "touch the ground",
    }

    # Run inference 
    action_chunk = np.array(policy.infer(observation)["actions"])

    #arm.set_servo_angle(angle=action_chunk[0,:6], is_radian=True)
    #arm.set_gripper_position(action_chunk[0,6])
    print(action_chunk)


""""
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
"""
