#!/usr/bin/env python3
"""
Materialize image_latent vectors to external .npy files and add a small pointer column.

For each episode_*.parquet:
  - For each row with a non-null image_latent, save it to:
        <latents_root>/<episode_basename>/frame_<idx>.npy
  - Add a new column 'latent_path' with the path string (or None if no latent).
  - Optionally clear the original image_latent column to shrink in-RAM usage.

Run this once, then update RandomHistoryDataset to load latents via latent_path
instead of touching the huge image_latent column inside the HF dataset.

Usage example:

  uv run scripts/materialize_latents_to_npy.py \\
      --data-dir ~/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/data/chunk-000 \\
      --latents-root ~/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/latents \\
      --clear-image-latent
"""

import argparse
import os
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm


def main() -> None:
    p = argparse.ArgumentParser(description="Export image_latent to .npy files and add latent_path pointers.")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing episode_*.parquet files (LeRobot shard, e.g. .../data/chunk-000).",
    )
    p.add_argument(
        "--latents-root",
        required=True,
        help="Root directory where latent .npy files will be stored.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=("float16", "float32"),
        help="Numeric dtype to store latents as (default float16 to halve disk + IO).",
    )
    p.add_argument(
        "--clear-image-latent",
        action="store_true",
        help="Set original image_latent cells to None after writing latent_path.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report counts but do not write any files.",
    )
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    latents_root = os.path.expanduser(args.latents_root)
    os.makedirs(latents_root, exist_ok=True)

    parquet_paths = sorted(
        p for p in pathlib.Path(data_dir).glob("episode_*.parquet") if p.is_file()
    )
    if not parquet_paths:
        raise SystemExit(f"No episode_*.parquet files found in {data_dir}")

    total_rows = 0
    total_with_latent = 0
    total_written = 0

    for p_path in tqdm(parquet_paths, desc="Episodes"):
        df = pd.read_parquet(p_path)
        n_rows = len(df)
        total_rows += n_rows

        if "image_latent" not in df.columns:
            continue

        base = p_path.stem  # e.g. episode_000012
        episode_dir = pathlib.Path(latents_root) / base
        if not args.dry_run:
            episode_dir.mkdir(parents=True, exist_ok=True)

        # Ensure latent_path column exists and is object dtype (can hold strings/None)
        if "latent_path" not in df.columns:
            df["latent_path"] = None
        if df["latent_path"].dtype != object:
            df["latent_path"] = df["latent_path"].astype(object)

        for idx in range(n_rows):
            val = df.at[idx, "image_latent"] if "image_latent" in df.columns else None
            if val is None:
                continue
            total_with_latent += 1

            # Force numeric dtype so we never write object arrays (which require allow_pickle=True to load).
            target_dtype = np.float16 if args.dtype == "float16" else np.float32
            latent_arr = np.asarray(val, dtype=target_dtype)
            if latent_arr.size == 0:
                continue

            rel_path = f"{base}/frame_{idx:06d}.npy"
            abs_path = episode_dir / f"frame_{idx:06d}.npy"

            if not args.dry_run:
                np.save(abs_path, latent_arr)
            df.at[idx, "latent_path"] = str(abs_path)
            total_written += 1

            if args.clear_image_latent:
                df.at[idx, "image_latent"] = None

        if not args.dry_run:
            df.to_parquet(p_path)

    print(f"Total rows scanned: {total_rows}")
    print(f"Rows with non-null image_latent: {total_with_latent}")
    print(f"Latent .npy files written / paths set: {total_written}")
    if args.dry_run:
        print("Dry run: no files or parquet modifications were written.")


if __name__ == "__main__":
    main()

