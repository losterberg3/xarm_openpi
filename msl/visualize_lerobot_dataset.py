#!/usr/bin/env python3
"""Visualize LeRobot dataset by creating videos for each task/primitive.

Usage:
    python scripts/visualize_lerobot_dataset.py --repo_id maggiewang/lego_primitives_01_30
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def main(repo_id: str, output_dir: str = "", num_episodes_per_task: int = 1, fps: int = 20):
    """Create visualization videos for a LeRobot dataset."""
    dataset_path = Path.home() / ".cache/huggingface/lerobot" / repo_id

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    # Set output directory
    if not output_dir:
        repo_name = repo_id.split("/")[-1]
        output_dir = f"data/libero/dataset_viz_{repo_name}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load episodes
    with open(dataset_path / "meta/episodes.jsonl") as f:
        episodes = [json.loads(l) for l in f]

    # Load tasks
    with open(dataset_path / "meta/tasks.jsonl") as f:
        tasks = {t["task_index"]: t["task"] for t in (json.loads(l) for l in f)}

    print(f"Dataset: {repo_id}")
    print(f"Tasks: {len(tasks)}")
    print(f"Episodes: {len(episodes)}")
    print(f"Output: {out_dir}")
    print()

    # Group episodes by task
    task_episodes = defaultdict(list)
    for ep in episodes:
        task = ep["tasks"][0]
        task_episodes[task].append(ep["episode_index"])

    print(f"Creating videos for each task ({num_episodes_per_task} episodes each)...")
    for task, ep_indices in task_episodes.items():
        safe_name = task.replace(" ", "_")
        task_dir = out_dir / safe_name
        task_dir.mkdir(exist_ok=True)

        # Save episodes as videos
        for i, ep_idx in enumerate(ep_indices[:num_episodes_per_task]):
            chunk = ep_idx // 1000
            parquet_path = dataset_path / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(parquet_path)

            frames = []
            for _, row in df.iterrows():
                img_data = row["image"]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                    frames.append(np.array(img))

            if frames:
                # Save as temp file with mp4v, then convert to h264 for compatibility
                temp_path = task_dir / f"episode_{i:03d}_temp.mp4"
                final_path = task_dir / f"episode_{i:03d}.mp4"

                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()

                # Convert to h264 for better compatibility
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', str(temp_path),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        str(final_path)
                    ], capture_output=True, check=True)
                    temp_path.unlink()  # Remove temp file
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # If ffmpeg fails, just rename temp to final
                    temp_path.rename(final_path)

                print(f"  {task}: episode {i} ({len(frames)} frames)")

    print(f"\nDone! Videos saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset")
    parser.add_argument("--repo_id", type=str, default="lars/xarm_history_exp_v2",
                        help="LeRobot dataset repo ID")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory for videos")
    parser.add_argument("--num_episodes_per_task", type=int, default=1,
                        help="Number of episodes to save per task")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    args = parser.parse_args()
    main(args.repo_id, args.output_dir, args.num_episodes_per_task, args.fps)