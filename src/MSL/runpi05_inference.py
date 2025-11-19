from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

#arm = XArmAPI('192.168.1.219')

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

#collect observations
"""
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    raise RuntimeError("Need at least two RealSense cameras connected")

# Extract serial numbers
serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
print("Found cameras:", serials)

pipelines = []
configs = []

for serial in serials:
    pipeline = rs.pipeline()
    config = rs.config()

    # 2. Bind this config to this specific device
    config.enable_device(serial)

    # 3. Enable streams (color + depth if you want)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # 4. Start pipeline
    pipeline.start(config)

    pipelines.append(pipeline)
    configs.append(config)

# 5. Get frames from BOTH cameras
framesA = pipelines[0].wait_for_frames()
#framesB = pipelines[1].wait_for_frames()

colorA = framesA.get_color_frame()
#colorB = framesB.get_color_frame()


# Configure depth and color streams
pipeline = rs.pipeline()
obs_config = rs.config()
"""

pipeline = rs.pipeline()
obs_config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = obs_config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

#obs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
obs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(obs_config)

frames = pipeline.wait_for_frames()
#depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

# Convert images to numpy arrays
#depth_image = np.asanyarray(depth_frame.get_data())
colorA_np = np.asanyarray(color_frame.get_data())
#colorA_np = np.asanyarray(colorA.get_data())
#colorB_np = np.asanyarray(colorB.get_data())

rgb_imageA = np.flip(colorA_np, axis=2)
#rgb_imageB = np.flip(colorB_np, axis=2)
a = rgb_imageA[np.newaxis, :] 
#b = rgb_imageB[np.newaxis, :] 
#a = cv2.resize(a, (224, 224))
#code, angles = arm.get_servo_angle(is_radian=True)
#state_nogrip = np.array(angles, dtype=np.float32)
state = np.zeros(6)
g_p = np.zeros(1)
#state[:7] = state_nogrip
a = np.zeros((224,224,3), dtype=np.float32)

observation = {
    "observation/exterior_image_1_left": a,
    "observation/wrist_image_left": a,
    "observation/gripper_position": g_p,
    "observation/joint_position": state,
    "prompt": "touch the ground"
}


# Run inference on a dummy example.

action_chunk = policy.infer(observation)["actions"]
print(action_chunk[:,:8])
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