from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.shared import array_typing as at
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples

class RandomHistoryDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, batch_size, window_size=5, *, action_horizon: int = 50):
        self.base = base_dataset
        self.window_size = window_size
        self.action_horizon = int(action_horizon)
        self.batch_size = batch_size
        self.half = batch_size // 2
        # Reach through possible wrapper stacks (TransformedDataset, TorchDataLoader wrappers, etc.)
        # to find the underlying LeRobot HF dataset.
        ds = base_dataset
        # Unwrap our own TransformedDataset wrappers
        while hasattr(ds, "_dataset"):
            ds = ds._dataset
        # Unwrap LeRobotDataset wrapper (it stores the HF dataset at .hf_dataset)
        self.hf_dataset = getattr(ds, "hf_dataset", None)
        if self.hf_dataset is None:
            # Some wrappers keep the dataset under .dataset
            inner = getattr(ds, "dataset", None)
            self.hf_dataset = getattr(inner, "hf_dataset", None) if inner is not None else None
        if self.hf_dataset is None:
            raise AttributeError(
                "RandomHistoryDataset: could not find underlying HuggingFace dataset. "
                f"Got base_dataset={type(base_dataset)}"
            )

        # IMPORTANT: LeRobot's torch formatting transform crashes on dict-typed image columns.
        # We do NOT call LeRobotDataset.__getitem__. Instead, we fetch raw HF rows and apply the
        # same TransformedDataset stack ourselves.
        self.hf_raw = self.hf_dataset.with_format(None)

        # Collect the wrapper transform stack so we can apply it ourselves.
        transforms: list[typing.Callable[[typing.Any], typing.Any]] = []
        ds_t = base_dataset
        while hasattr(ds_t, "_transform") and hasattr(ds_t, "_dataset"):
            transforms.append(ds_t._transform)  # type: ignore[attr-defined]
            ds_t = ds_t._dataset  # type: ignore[attr-defined]
        self._transform_stack = transforms  # outer->inner

        # Fast latent-availability mask for HISTORY sampling only.
        # Prefer latent_path (small string column) if present, else fall back to image_latent != None.
        self._has_latent = None
        try:
            cols = set(getattr(self.hf_raw, "column_names", []))
        except Exception:
            cols = set()
        if "latent_path" in cols:
            paths = self.hf_raw["latent_path"]
            self._has_latent = np.array([(p is not None) and (p != "") for p in paths], dtype=bool)
        elif "image_latent" in cols:
            lats = self.hf_raw["image_latent"]
            self._has_latent = np.array([v is not None for v in lats], dtype=bool)

        all_pos = np.where(np.array(self.hf_raw["is_decision"]) == 1)[0]
        all_neg = np.where(np.array(self.hf_raw["is_decision"]) == 0)[0]
        self.pos_indices = [
            idx for idx in all_pos 
            if self.hf_raw[int(idx)]["index"] >= self.window_size
        ]
        self.neg_indices = [
            idx for idx in all_neg 
            if self.hf_raw[int(idx)]["index"] >= self.window_size
        ]
        # NOTE: Do NOT filter current-frame pools by latent availability.
        # The latent is only used for history; current frame uses observation+actions.
        self.pos_indices = np.array(self.pos_indices)
        self.neg_indices = np.array(self.neg_indices)
        logging.info(f"Initialized GRU Dataset: {len(self.pos_indices)} positive, {len(self.neg_indices)} negative samples.")

    def __len__(self):
        return len(self.base)

    def _get_frame_with_history(self, idx, is_decision):
        idx_int = int(idx)  # HF datasets don't accept numpy scalar ints
        raw = self.hf_raw[idx_int]
        if not isinstance(raw, dict):
            raw = dict(raw)

        # Decode common LeRobot image dicts {bytes,path} to uint8 HWC arrays.
        def _decode_image_cell(cell):
            try:
                if isinstance(cell, dict) and cell.get("bytes") is not None:
                    import io
                    from PIL import Image

                    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            except Exception:
                pass
            return cell

        for k in ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"):
            if k in raw:
                raw[k] = _decode_image_cell(raw[k])

        # Build action sequence (T, action_dim) from per-frame actions in the HF dataset.
        ep = raw.get("episode_index")
        actions_seq = []
        last = None
        for t in range(self.action_horizon):
            j = idx_int + t
            if j >= len(self.hf_raw):
                break
            rj = self.hf_raw[int(j)]
            if not isinstance(rj, dict):
                rj = dict(rj)
            if ep is not None and rj.get("episode_index") != ep:
                break
            a = np.asarray(rj.get("actions"))
            if a.ndim == 2 and a.shape[0] == 1:
                a = a[0]
            last = a
            actions_seq.append(a)
        if not actions_seq:
            raise ValueError(f"RandomHistoryDataset: no actions found for idx={idx_int}")
        while len(actions_seq) < self.action_horizon:
            actions_seq.append(last)
        raw["actions"] = np.stack(actions_seq, axis=0)

        # Apply transforms inner->outer.
        sample: typing.Any = raw
        for t in reversed(self._transform_stack):
            sample = t(sample)
        if not isinstance(sample, dict) or "actions" not in sample:
            raise ValueError(
                "RandomHistoryDataset: expected transformed sample to be a dict with 'actions'. "
                f"Got type={type(sample)} keys={list(sample.keys()) if isinstance(sample, dict) else None}"
            )
        action = sample["actions"]
        frame_row = self.hf_raw[idx_int]
        frame_in_ep = frame_row["index"]
        start_of_ep = idx_int - int(frame_in_ep)
        past_indices = np.arange(start_of_ep, idx_int)

        sig_values = np.array(self.hf_raw[past_indices]["significance"])
        target_val = 1 if is_decision else 0
        valid_pool = past_indices[sig_values == target_val]
        # Ensure sampled history frames have latents available.
        if self._has_latent is not None and len(valid_pool):
            valid_pool = valid_pool[self._has_latent[valid_pool]]
        if len(valid_pool) == 0:
            selected = np.full(self.window_size, idx_int, dtype=np.int64)
        else:
            selected = np.random.choice(valid_pool, self.window_size, replace=True)
        selected.sort()

        def _load_latent(row_idx: int) -> np.ndarray:
            row = self.hf_raw[int(row_idx)]
            # Preferred: external pointer produced by materialize_latents_to_npy.py
            latent_path = row.get("latent_path") if isinstance(row, dict) and "latent_path" in row else None
            if latent_path:
                arr = np.load(latent_path, allow_pickle=True)
                # If older files were saved as object arrays, coerce to float32.
                if getattr(arr, "dtype", None) == object:
                    arr = np.asarray(arr.tolist(), dtype=np.float32)
                arr = np.asarray(arr)
            # Fallback: inline image_latent (original behavior)
            else:
                lat = row["image_latent"]
                arr = np.asarray(lat)
                if arr.shape == () and lat is None:
                    raise ValueError(f"missing latent for row {row_idx} (no latent_path and image_latent is None)")

            # Normalize shape to (256, 2048). Some files may be (1,256,2048) or flat.
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 1 and arr.size == 256 * 2048:
                arr = arr.reshape(256, 2048)
            if arr.shape != (256, 2048):
                raise ValueError(f"latent has unexpected shape {arr.shape} for row {row_idx} (path={latent_path!r})")
            return arr.astype(np.float32, copy=False)

        latents = np.stack([_load_latent(i) for i in selected], axis=0)
        
        # Keep full transformed sample (includes observation keys + actions), and add latents.
        return {**sample, "actions": action, "latents": latents}

    def __getitem__(self, batch_idx):
        slot_in_batch = batch_idx % self.batch_size
        
        if slot_in_batch < self.half:
            idx = int(np.random.choice(self.pos_indices))
            return self._get_frame_with_history(idx, is_decision=True)
        else:
            idx = int(np.random.choice(self.neg_indices))
            return self._get_frame_with_history(idx, is_decision=False)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions, at.Array]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        gru=config.model.gru,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    gru: bool = False,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions, at.Array]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    if gru:
        logging.info("Wrapping dataset with RandomHistoryDataset for GRU training.")
        dataset = RandomHistoryDataset(dataset, batch_size=batch_size, window_size=10, action_horizon=action_horizon)
        shuffle = False

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            if "latents" in batch:
                yield _model.Observation.from_dict(batch), batch["actions"], batch.get("latents")
            else:
                yield _model.Observation.from_dict(batch), batch["actions"]
