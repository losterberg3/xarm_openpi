import pandas as pd
import cv2
import numpy as np
from PIL import Image
import io

# 1. Load your specific parquet file
parquet_path = "~/.cache/huggingface/lerobot/lars/xarm_history_exp_v2/data/chunk-000/episode_000009.parquet"
df = pd.read_parquet(parquet_path)

# 2. Extract frames
frames = []
for _, row in df.iterrows():
    # 'image' is the standard LeRobot column name
    img_data = row["exterior_image_1_left"] 
    img = Image.open(io.BytesIO(img_data["bytes"]))
    frames.append(np.array(img))

# 3. Write to Video
h, w = frames[0].shape[:2]
fps = 10
out = cv2.VideoWriter('000009.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

for frame in frames:
    # Convert RGB to BGR for OpenCV
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
print("Video saved as single_demo.mp4")