import numpy as np
import pandas as pd
import io
from PIL import Image
import os
import glob
from tqdm import tqdm

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as _config

config = _config.get_config("pi05_xarm")
checkpoint_dir = download.maybe_download("/home/larsosterberg/msl/openpi/checkpoints/pi05_xarm_finetune/lars_history_exp_v1/25000")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir, latents_out=True)

# --- Path Config ---
base_path = os.path.expanduser("~/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/data/chunk-000/")
parquet_files = sorted(glob.glob(os.path.join(base_path, "episode_0000*.parquet")))
# Target episodes 00 through 33
parquet_files = [f for f in parquet_files if 0 <= int(os.path.basename(f).split('_')[1].split('.')[0]) <= 33]

for p_path in parquet_files:
    filename = os.path.basename(p_path)
    df = pd.read_parquet(p_path)

    if "is_decision" not in df.columns or not (df["is_decision"] == 1).any():
        print(f"Skipping {filename}: No 'is_decision' markers found.")
        continue

    first_decision_idx = df.index[df["is_decision"] == 1][0]
    print(f"\n>>> Processing {filename} (0 to {first_decision_idx})")

    if "image_latent" not in df.columns:
        df["image_latent"] = None
        df["image_latent"] = df["image_latent"].astype(object)

    for current_idx in tqdm(range(0, first_decision_idx)):
        ext_bytes = df.at[current_idx, "exterior_image_1_left"]["bytes"]
        wrist_bytes = df.at[current_idx, "wrist_image_left"]["bytes"]
        
        ext_img = np.array(Image.open(io.BytesIO(ext_bytes)).convert("RGB"))
        wrist_img = np.array(Image.open(io.BytesIO(wrist_bytes)).convert("RGB"))

        gripper_val = df.at[current_idx, "gripper_position"]

        obs = {
            "observation/exterior_image_1_left": ext_img,
            "observation/wrist_image_left": wrist_img,
            "observation/gripper_position": np.array([gripper_val], dtype=np.float32),
            "observation/joint_position": np.array(df.at[current_idx, "joint_position"], dtype=np.float32),
            "prompt": "Grab the yellow bottle and place it on the pink marker", # Use your exact training prompt
        }

        print(f"\n[DEBUG] Frame {current_idx} Inspection:")
        
        for key, value in obs.items():
            if value is None:
                print(f"!! CRITICAL: {key} is None")
            elif hasattr(value, 'shape'):
                print(f"  {key} shape: {value.shape} | dtype: {value.dtype}")
            elif isinstance(value, (list, str)):
                print(f"  {key}: {value[:50]}...") # Print first bit of prompt/list
            else:
                print(f"  {key} type: {type(value)}")

        try:
            print(f"Querying raw model for latents...")
            raw_inference = policy.infer(obs)
            
            if "latents" in raw_inference:
                latents = raw_inference["latents"]
                latents_np = np.array(latents)
                
                if latents_np.ndim == 3:
                    latents_np = latents_np[0]
   
                df.at[current_idx, "image_latent"] = latents_np.tolist()
                print(f"Success! Latent saved for frame {current_idx}")
            else:
                print(f"!! Latents missing. Keys: {raw_inference.keys()}")
                break
                
        except Exception as e:
            print(f"!!! INFERENCE CRASHED at index {current_idx} !!!")
            print(f"Error Message: {e}")
            import traceback
            traceback.print_exc() 
            break

    df.to_parquet(p_path)
    print(f"Done: {filename}")