"""Convert a quaternion-action LeRobot dataset to EEF (Euler) actions in place.

All repos are under HF_LEROBOT_HOME. Converts parquet files in place (one episode at a time,
temp file then replace) so no extra full-dataset copy is made. Updates meta/info.json to set
actions shape to (7,).

Usage:
    python scripts/convert_dataset.py

Optional: --dry-run to only print what would be converted and show first-row sanity check.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from lerobot.common.constants import HF_LEROBOT_HOME


SRC_REPO = "lars/xarm_history_exp_v2"
EEF_REPO = "lars/xarm_history_exp_v1"


def quat_to_euler_rad(quat: np.ndarray) -> np.ndarray:
    """[qx, qy, qz, qw] -> euler rad (3,) with roll/yaw in [0, 360)° then to rad."""
    quat = np.asarray(quat, dtype=np.float32).ravel()
    if quat.size != 4:
        raise ValueError(f"Expected 4 elements (quat), got {quat.size}")
    euler_deg = Rotation.from_quat(quat).as_euler("xyz", degrees=True).astype(np.float32)
    euler_deg[0] = euler_deg[0] % 360
    euler_deg[2] = euler_deg[2] % 360
    return (euler_deg * np.pi / 180).astype(np.float32)


def quat_to_eef_action(a: np.ndarray) -> np.ndarray:
    """Convert one action row [x, y, z, qx, qy, qz, qw, grip] -> [x, y, z, rx, ry, rz, grip]."""
    a = np.asarray(a, dtype=np.float32).ravel()
    if a.size != 8:
        raise ValueError(f"Expected 8 elements, got {a.size}")
    pos = a[:3]
    grip = a[7]
    euler_rad = quat_to_euler_rad(a[3:7])
    return np.concatenate([pos, euler_rad, [grip]]).astype(np.float32)


def quat_to_eef_position(e: np.ndarray) -> np.ndarray:
    """Convert eef_position [x, y, z, qx, qy, qz, qw] -> [x, y, z, rx, ry, rz] (6 elements)."""
    e = np.asarray(e, dtype=np.float32).ravel()
    if e.size != 7:
        raise ValueError(f"Expected 7 elements (eef_position), got {e.size}")
    pos = e[:3]
    euler_rad = quat_to_euler_rad(e[3:7])
    return np.concatenate([pos, euler_rad]).astype(np.float32)


def convert_parquet_in_place(p_path: Path, dry_run: bool) -> int:
    """Convert actions and eef_position columns (quat -> euler) in one episode parquet. Returns row count."""
    df = pd.read_parquet(p_path)
    n_rows = len(df)
    if n_rows == 0:
        return 0

    if "actions" in df.columns:
        a0 = np.asarray(df["actions"].iloc[0], dtype=np.float32).ravel()
        if a0.size == 8:
            new_actions = []
            for i in range(n_rows):
                a = np.asarray(df["actions"].iloc[i], dtype=np.float32).ravel()
                new_actions.append(quat_to_eef_action(a))
            df["actions"] = new_actions

    if "eef_position" in df.columns:
        e0 = np.asarray(df["eef_position"].iloc[0], dtype=np.float32).ravel()
        if e0.size == 7:
            new_eef = []
            for i in range(n_rows):
                e = np.asarray(df["eef_position"].iloc[i], dtype=np.float32).ravel()
                new_eef.append(quat_to_eef_position(e))
            df["eef_position"] = new_eef
    if dry_run:
        return n_rows
    tmp = p_path.with_suffix(p_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p_path)
    return n_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Convert quat actions to EEF (Euler) in place.")
    p.add_argument("--dry-run", action="store_true", help="Do not write; only sanity check and list files.")
    args = p.parse_args()

    base = Path(HF_LEROBOT_HOME)
    src_path = base / SRC_REPO
    eef_path = base / EEF_REPO
    data_dir = src_path / "data"
    meta_dir = src_path / "meta"

    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset not found at {src_path}")

    # Sanity: first row of source (from first parquet we find)
    chunk_dirs = sorted(data_dir.glob("chunk-*")) if data_dir.exists() else []
    if not chunk_dirs:
        raise FileNotFoundError(f"No data/chunk-* directories under {src_path}")
    first_parquet = next((f for d in chunk_dirs for f in sorted(d.glob("episode_*.parquet"))), None)
    if first_parquet is None:
        raise FileNotFoundError(f"No episode_*.parquet under {src_path}/data")
    df0 = pd.read_parquet(first_parquet)
    print("\n=== Sanity: first row of SRC ===")
    if "actions" in df0.columns:
        a_src = np.asarray(df0["actions"].iloc[0], dtype=np.float32).ravel()
        print("actions:", a_src)
        if a_src.size == 8:
            print("converted actions (7):", quat_to_eef_action(a_src))
    if "eef_position" in df0.columns:
        e_src = np.asarray(df0["eef_position"].iloc[0], dtype=np.float32).ravel()
        print("eef_position:", e_src)
        if e_src.size == 7:
            print("converted eef_position (6):", quat_to_eef_position(e_src))

    # EEF comparison
    try:
        if eef_path.exists():
            eef_data = eef_path / "data"
            eef_chunks = sorted(eef_data.glob("chunk-*")) if eef_data.exists() else []
            eef_first = next((f for d in eef_chunks for f in sorted(d.glob("episode_*.parquet"))), None)
            if eef_first is not None:
                edf = pd.read_parquet(eef_first)
                a_eef = np.asarray(edf["actions"].iloc[0], dtype=np.float32).ravel()
                print("\n=== First row of EEF_REPO (local) ===")
                print("raw actions:", a_eef)
                print("position (xyz):", a_eef[:3])
                print("Euler (rad, xyz):", a_eef[3:6])
                print("Euler (deg, xyz):", np.rad2deg(a_eef[3:6]))
        else:
            print(f"\nEEF_REPO path not found at {eef_path}, skipping comparison.")
    except Exception as e:
        print(f"\nWarning: could not load EEF_REPO for comparison: {e}")

    if args.dry_run:
        print("\n[DRY RUN] Would convert the following parquet files:")
        total = 0
        for chunk_dir in chunk_dirs:
            files = sorted(chunk_dir.glob("episode_*.parquet"))
            for f in files:
                print(f"  {f.relative_to(src_path)}")
                total += 1
        print(f"Total: {total} episode parquets. Run without --dry-run to convert in place.")
        return

    # Convert each parquet in place
    total_rows = 0
    for chunk_dir in tqdm(chunk_dirs, desc="Chunks"):
        files = sorted(chunk_dir.glob("episode_*.parquet"))
        for p_path in tqdm(files, desc=chunk_dir.name, leave=False):
            total_rows += convert_parquet_in_place(p_path, dry_run=False)

    print(f"\nConverted {total_rows} rows in place.")

    # Update meta/info.json: actions 8 -> 7, eef_position 7 -> 6
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        changed = False
        if "features" in info and "actions" in info["features"]:
            old = info["features"]["actions"].get("shape", [8])
            info["features"]["actions"]["shape"] = [7]
            print(f"Updated {info_path}: actions shape {old} -> [7]")
            changed = True
        if "features" in info and "eef_position" in info["features"]:
            old = info["features"]["eef_position"].get("shape", [7])
            info["features"]["eef_position"]["shape"] = [6]
            print(f"Updated {info_path}: eef_position shape {old} -> [6]")
            changed = True
        if changed:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
        elif "features" not in info:
            print(f"Note: {info_path} has no features; you may need to set actions/eef_position shapes manually.")
    else:
        print(f"Note: {info_path} not found; you may need to set actions and eef_position shapes in meta manually.")

    # Update meta/episode_stats.jsonl: if it tracks action_dim, switch 8 -> 7.
    stats_path = meta_dir / "episodes_stats.jsonl"
    if stats_path.exists():
        lines = []
        changed = False
        with open(stats_path) as f:
            for line in f:
                if not line.strip():
                    lines.append(line)
                    continue
                rec = json.loads(line)
                if "action_dim" in rec and rec["action_dim"] == 8:
                    rec["action_dim"] = 7
                    changed = True
                lines.append(json.dumps(rec) + "\n")
        if changed:
            with open(stats_path, "w") as f:
                f.writelines(lines)
            print(f"Updated {stats_path}: set action_dim from 8 to 7 where present.")
        else:
            print(f"Note: {stats_path} has no action_dim==8 entries to update.")
    else:
        print(f"Note: {stats_path} not found; you may need to adjust episode stats manually if they encode action_dim.")

    print("Done.")


if __name__ == "__main__":
    main()
