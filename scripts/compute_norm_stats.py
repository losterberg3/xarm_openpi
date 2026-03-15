"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

LeRobot datasets store image columns as dicts (bytes/path). The default torch format
then does torch.tensor(dict) and raises "Could not infer dtype of dict". Norm stats
only need state and actions, so we fall back to loading only those columns when that
happens.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


def _compute_norm_stats_numeric_only(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None,
) -> dict[str, normalize.RunningStats]:
    """Compute state/action stats from LeRobot using only numeric columns (no images)."""
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    ds = dataset
    while hasattr(ds, "_dataset"):
        ds = ds._dataset
    if not hasattr(ds, "hf_dataset"):
        raise RuntimeError("Numeric-only path requires LeRobot dataset with hf_dataset")
    hf = ds.hf_dataset
    cols = set(hf.column_names)
    if "joint_position" in cols and "gripper_position" in cols:
        state_cols = ["joint_position", "gripper_position"]
    elif "eef_position" in cols and "gripper_position" in cols:
        state_cols = ["eef_position", "gripper_position"]
    else:
        raise ValueError("Dataset must have (joint_position + gripper_position) or (eef_position + gripper_position)")
    need = state_cols + ["actions"] + [c for c in ("episode_index", "index") if c in cols]
    slim = hf.select_columns(need).with_format(None)
    n = len(slim)
    action_horizon = int(action_horizon)
    delta_mask = transforms.make_bool_mask(6, -1)
    delta_fn = transforms.DeltaActions(delta_mask)
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    for idx in tqdm.tqdm(range(n), desc="Computing stats (numeric only)"):
        if max_frames is not None and idx >= max_frames:
            break
        row = slim[int(idx)]
        ep = row.get("episode_index")
        ep = int(ep.item()) if ep is not None and hasattr(ep, "item") else int(ep) if ep is not None else 0
        state_parts = [np.asarray(row[k]).ravel() for k in state_cols]
        state = np.concatenate(state_parts).astype(np.float32)
        actions_list = [np.asarray(row["actions"]).ravel().astype(np.float32)]
        for j in range(1, action_horizon):
            i2 = idx + j
            if i2 >= n:
                break
            r2 = slim[int(i2)]
            ep2 = r2.get("episode_index")
            ep2 = int(ep2.item()) if ep2 is not None and hasattr(ep2, "item") else int(ep2) if ep2 is not None else 0
            if ep2 != ep:
                break
            actions_list.append(np.asarray(r2["actions"]).ravel().astype(np.float32))
        while len(actions_list) < action_horizon:
            actions_list.append(actions_list[-1].copy())
        actions_chunk = np.stack(actions_list, axis=0)
        data = delta_fn({"state": state, "actions": actions_chunk})
        stats["state"].update(data["state"][np.newaxis])
        stats["actions"].update(data["actions"][np.newaxis])
    return stats


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))
    else:
        try:
            data_loader, num_batches = create_torch_dataloader(
                data_config,
                config.model.action_horizon,
                config.batch_size,
                config.model,
                config.num_workers,
                max_frames,
            )
            for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
                for key in keys:
                    stats[key].update(np.asarray(batch[key]))
        except RuntimeError as e:
            err = str(e) + (str(e.__cause__) if e.__cause__ else "")
            if "Could not infer dtype of dict" not in err:
                raise
            print("Falling back to numeric-only loading (LeRobot image columns are dicts).")
            stats = _compute_norm_stats_numeric_only(
                data_config,
                config.model.action_horizon,
                config.batch_size,
                config.model,
                max_frames,
            )

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
