import pandas as pd
import os

path = os.path.expanduser("~/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/data/chunk-000/episode_000010.parquet")

if os.path.exists(path):
    df = pd.read_parquet(path)

    has_sig = "significance" in df.columns
    has_dec = "is_decision" in df.columns
    has_lat = "image_latent" in df.columns
    
    print(f"--- Inspection of {os.path.basename(path)} ---")
    print(f"Columns present: Significance: {has_sig}, Decision: {has_dec}, Latent: {has_lat}")
    
    if has_sig and len(df) > 100:
        val_sig = df.at[100, "significance"]
        val_dec = df.at[100, "is_decision"]
        val_lat = df.at[100, "image_latent"]
        print(f"Values at index 100 -> Significance: {val_sig}, Is_Decision: {val_dec}")
        print(val_lat)
    else:
        print("Index 100 not found or columns not initialized yet.")
else:
    print("File not found.")