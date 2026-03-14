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

config = _config.get_config("pi05_xarm_gru")
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_gru_addition/pytorch/14999")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

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
    # Prefer VLAEXPLAIN_PATH; then try msl sibling dir (openpi and VLAExplain both under msl/)
    vlaexplain_path = os.environ.get("VLAEXPLAIN_PATH", "").strip()
    if not vlaexplain_path or not os.path.isdir(vlaexplain_path):
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
        print(f"Attention streaming enabled on port {args.attention_viewer_port}.")
        print(
            f"  From your LOCAL machine (if SSH): run:\n"
            f"    ssh -L {args.attention_viewer_port}:localhost:{args.attention_viewer_port} <user>@<remote-host>\n"
            f"  Then open in browser: http://localhost:{args.attention_viewer_port}"
        )
    except ImportError as e:
        print(
            f"Could not enable attention streaming: {e}\n"
            "Install gradio (e.g. uv pip install gradio) and set VLAEXPLAIN_PATH to the VLAExplain repo root, "
            "or run inference_vis from a directory where VLAExplain is at ../VLAExplain."
        )

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

    observation = {
        "observation/exterior_image_1_left": b,
        "observation/wrist_image_left": a,
        "observation/gripper_position": np.zeros(1),
        "observation/joint_position": np.zeros(6),
        "prompt": "Drop the block in the cup and then knock that same cup over",
    }
    return observation

history = None
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
    inference = policy.infer(observation, history=history)
    _inference_step += 1
    history = inference.get("history", None)
    

    

    