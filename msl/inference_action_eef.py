import argparse
import os
import sys
import threading
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
ACTION_ROLLOUT = 20

parser = argparse.ArgumentParser()
parser.add_argument(
    "--stream-attention",
    action="store_true",
    help="Stream attention to VLAExplain viewer (requires PyTorch model and VLAExplain repo). "
    "Use PyTorch checkpoint (model.safetensors); see STREAMING.md.",
)
parser.add_argument("--attention-viewer-port", type=int, default=7863, help="Port for attention viewer (default 7863)")
args, _ = parser.parse_known_args()

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
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_xarm_finetune/lars_eef_2_11/20000")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir, language_out=False)

# Optional: stream attention to VLAExplain (only works with PyTorch model)
def _setup_attention_stream():
    if not args.stream_attention:
        return
    is_pytorch = getattr(policy, "_is_pytorch_model", False)
    if not is_pytorch:
        print(
            "Warning: --stream-attention requires a PyTorch checkpoint (model.safetensors). "
            "Convert with: python examples/convert_jax_model_to_pytorch.py --checkpoint_dir <jax_ckpt> --output_path <out>"
        )
        return
    vlaexplain_path = os.environ.get("VLAEXPLAIN_PATH")
    if not vlaexplain_path or not os.path.isdir(vlaexplain_path):
        # From openpi/msl/ -> ../../../ = repo parent (e.g. msl/), then VLAExplain
        vlaexplain_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "VLAExplain")
    if not os.path.isdir(vlaexplain_path):
        vlaexplain_path = os.path.join(os.path.dirname(os.path.abspath(str(checkpoint_dir))), "..", "..", "..", "VLAExplain")
    if os.path.isdir(vlaexplain_path) and vlaexplain_path not in sys.path:
        sys.path.insert(0, vlaexplain_path)
    try:
        from stream_attention_viewer import push_attention, run_gradio_ui
        import openpi.models_pytorch.gemma_pytorch as _gemma_mod
        def _callback(images: dict, expert_attn: dict):
            push_attention(
                step=_gemma_mod.ATTENTION_STREAM_STEP,
                raw_images={k: v for k, v in images.items() if k in ("image1", "image2")},
                expert_attn=expert_attn,
                time_step=0,
            )
        _gemma_mod.ATTENTION_STREAM_CALLBACK = _callback
        threading.Thread(target=run_gradio_ui, kwargs={"port": args.attention_viewer_port}, daemon=True).start()
        print(f"Attention streaming enabled. Open http://localhost:{args.attention_viewer_port}")
    except ImportError as e:
        print(f"Could not enable attention streaming: {e}. Install gradio and set VLAEXPLAIN_PATH to the VLAExplain repo.")

_setup_attention_stream()

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
        "prompt": "Grab the yellow bottle and place it on the pink marker",
    }
    return observation

def interpolate_action(state, goal):
    delta_increment = (goal - state) / (DT * CONTROL_HZ * 6)

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
       
_inference_step = 0
while True:

    observation = get_observation()

    # If attention streaming is enabled (PyTorch only), set current images and step for the model callback
    if args.stream_attention and getattr(policy, "_is_pytorch_model", False):
        try:
            import openpi.models_pytorch.gemma_pytorch as _gemma_mod
            if _gemma_mod.ATTENTION_STREAM_CALLBACK is not None:
                a = observation.get("observation/exterior_image_1_left")
                b = observation.get("observation/wrist_image_left")
                if a is not None and b is not None:
                    a = np.asarray(a)
                    b = np.asarray(b)
                    if a.ndim == 3 and a.shape[0] == 3:
                        a = np.transpose(a, (1, 2, 0))
                    if b.ndim == 3 and b.shape[0] == 3:
                        b = np.transpose(b, (1, 2, 0))
                    _gemma_mod.ATTENTION_STREAM_IMAGES = {"image1": a, "image2": b}
                    _gemma_mod.ATTENTION_STREAM_STEP = _inference_step
        except Exception:
            pass

    print("Running inference")
    inference = policy.infer(observation)
    _inference_step += 1
    
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
        #print("command")
        #print(cmd_joint_pose)
        cmd_gripper_pose = (action[count,6]) * -860 + 850 # unnormalize the gripper action
        arm.set_gripper_position(cmd_gripper_pose)

        count += 1
        time_left = DT - (time.perf_counter() - t0)
        
        time.sleep(max(time_left,0))
    