import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image
import io

parquet_path = "~/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/data/chunk-000/episode_000001.parquet"
df = pd.read_parquet(parquet_path)

frames = []
all_states = []
all_actions = []

for _, row in df.iterrows():
    img_data = row["exterior_image_1_left"]
    img = Image.open(io.BytesIO(img_data["bytes"]))
    frames.append(np.array(img))

    curr_state = np.concatenate([row["joint_position"], [row["gripper_position"]]])
    all_states.append(curr_state)

    all_actions.append(row["actions"])

state_tensor = torch.tensor(np.array(all_states), dtype=torch.float32)
action_tensor = torch.tensor(np.array(all_actions), dtype=torch.float32)

h, w = frames[0].shape[:2]
out = cv2.VideoWriter('000009.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
for f in frames:
    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
out.release()

torch.save({
    'states': state_tensor,
    'actions': action_tensor
}, '000009_telemetry.pt')

print(f"Exported {len(frames)} frames and telemetry with state dim {state_tensor.shape[1]}")