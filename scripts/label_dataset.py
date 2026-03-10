import pandas as pd
import os
import io
import glob
from PIL import Image

base_path = os.path.expanduser("~/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/data/chunk-000/")
parquet_files = sorted(glob.glob(os.path.join(base_path, "episode_0000*.parquet")))
parquet_files = [f for f in parquet_files if 0 <= int(os.path.basename(f).split('_')[1].split('.')[0]) <= 33]

tmp_view = "current_labeling_frame.png"
SAVE_INTERVAL = 50 

for p_path in parquet_files:
    filename = os.path.basename(p_path)
    df = pd.read_parquet(p_path)
    
    if "significance" not in df.columns: df["significance"] = 0
    if "is_decision" not in df.columns: df["is_decision"] = 0

    idx = 0
    active_mode = "0" # "0", "1", or "d"
    
    while idx < len(df):
        row = df.iloc[idx]
        Image.open(io.BytesIO(row["exterior_image_1_left"]["bytes"])).save(tmp_view)
        
        sig, dec = df.at[idx, "significance"], df.at[idx, "is_decision"]
        mode_str = "SIGNIFICANT (1)" if active_mode == "1" else "DECISION (d)" if active_mode == "d" else "UNIMPORTANT (0)"
        
        print(f"\n[{filename}] Frame: {idx}/{len(df)} | Current Sig: {sig} Dec: {dec}")
        print(f"*** ACTIVE MODE: {mode_str} ***")
        cmd = input("Cmd (1, d, 0, #, f, b, q): ").lower().strip()
        
        if cmd in ['1', 'd', '0']:
            active_mode = cmd
            if active_mode == '1':
                df.at[idx, "significance"] = 1
            elif active_mode == 'd':
                df.at[idx, "is_decision"] = 1
            elif active_mode == '0':
                df.at[idx, "significance"] = 0
                df.at[idx, "is_decision"] = 0
            idx += 1

        elif cmd.isdigit():
            count = int(cmd)
            end_idx = min(len(df), idx + count)
            if active_mode == '1':
                df.loc[idx:end_idx-1, "significance"] = 1
            elif active_mode == 'd':
                df.loc[idx:end_idx-1, "is_decision"] = 1
            elif active_mode == '0':
                df.loc[idx:end_idx-1, "significance"] = 0
            idx = end_idx

        elif cmd.startswith('f'):
            skip = int(cmd[1:]) if len(cmd) > 1 else 20
            idx = min(len(df) - 1, idx + skip)

        elif cmd == 'b':
            idx = max(0, idx - 1)
        elif cmd == 'q':
            df.to_parquet(p_path)
            print(f"\nSaved {filename}. Quitting.")
            exit()
        else:
            if active_mode == '1': df.at[idx, "significance"] = 1
            elif active_mode == 'd': df.at[idx, "is_decision"] = 1
            idx += 1
            
        if idx > 0 and idx % SAVE_INTERVAL == 0:
            df.to_parquet(p_path)

    df.to_parquet(p_path)
    print(f"\nFinished {filename}")