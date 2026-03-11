import json
import os
import shutil

info_path = os.path.expanduser("~/.cache/huggingface/lerobot/lars/xarm_history_exp_v1/meta/info.json")
backup_path = info_path + ".bak"

def update_metadata():
    if not os.path.exists(info_path):
        print(f"Error: Could not find {info_path}")
        return

    shutil.copy2(info_path, backup_path)
    print(f"Backup created at {backup_path}")

    with open(info_path, 'r') as f:
        info = json.load(f)

    new_features = {
        "significance": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "is_decision": {
            "dtype": "int64",
            "shape": [1],
            "names": None
        },
        "image_latent": {
            "dtype": "float32", 
            "shape": [256, 2048], 
            "names": None
        }
    }

    if "features" not in info:
        print("Error: 'features' key not found in info.json")
        return

    added = False
    for feat_name, feat_def in new_features.items():
        if feat_name not in info["features"]:
            info["features"][feat_name] = feat_def
            print(f"Adding feature: {feat_name}")
            added = True
        else:
            print(f"Feature '{feat_name}' already exists. Skipping.")

    if added:
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Successfully updated {info_path}")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    update_metadata()