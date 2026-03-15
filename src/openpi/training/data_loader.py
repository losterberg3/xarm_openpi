from collections.abc import Iterator, Sequence
import io
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
import torchvision.transforms as T

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.shared import array_typing as at
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

# PIL is used for image-dict decoding in the safe HF transform.
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # type: ignore[misc, assignment]


def _decode_image_dict_to_tensor(d: typing.Any) -> torch.Tensor:
    """Decode a HuggingFace image dict (path/bytes) to a torch tensor (C, H, W) float [0,1]."""
    if not isinstance(d, dict) or PILImage is None:
        return torch.tensor(d)
    path, bytes_ = d.get("path"), d.get("bytes")
    if bytes_ is not None:
        img = PILImage.open(io.BytesIO(bytes_)).convert("RGB")
    elif path is not None and os.path.isfile(path):
        img = PILImage.open(path).convert("RGB")
    else:
        raise ValueError(f"Image dict has no loadable path or bytes: {list(d.keys())}")
    return T.functional.to_tensor(img)


def _safe_hf_transform_to_torch(items_dict: dict) -> dict:
    """Like LeRobot's hf_transform_to_torch but never calls torch.tensor() on dicts.

    Image columns in LeRobot/HF are often stored as dicts (path/bytes). The default
    transform does torch.tensor(dict) and raises 'Could not infer dtype of dict'.
    Here we decode image dicts to tensors and leave other dict columns unchanged.
    """
    to_tensor = T.ToTensor()
    for key in items_dict:
        vals = items_dict[key]
        first = vals[0] if vals else None
        if isinstance(first, torch.Tensor):
            continue
        if PILImage is not None and isinstance(first, PILImage.Image):
            items_dict[key] = [to_tensor(img) for img in vals]
        elif first is None:
            pass
        elif isinstance(first, dict):
            out = []
            for x in vals:
                if isinstance(x, dict) and ("path" in x or "bytes" in x):
                    out.append(_decode_image_dict_to_tensor(x))
                elif isinstance(x, str) or isinstance(x, dict):
                    out.append(x)
                else:
                    out.append(torch.tensor(x))
            items_dict[key] = out
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in vals]
    return items_dict


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

def _load_latent_from_row(hf_raw: typing.Any, row_idx: int) -> np.ndarray:
    """Load latent for one row from hf_raw (latent_path or image_latent). Shape (256, 2048)."""
    row = hf_raw[int(row_idx)]
    row = row if isinstance(row, dict) else dict(row)
    latent_path = row.get("latent_path")
    if latent_path:
        arr = np.load(latent_path, allow_pickle=True)
        if getattr(arr, "dtype", None) == object:
            arr = np.asarray(arr.tolist(), dtype=np.float32)
        arr = np.asarray(arr)
    else:
        lat = row.get("image_latent")
        if lat is None:
            raise ValueError(f"missing latent for row {row_idx} (no latent_path and image_latent is None)")
        arr = np.asarray(lat)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and arr.size == 256 * 2048:
        arr = arr.reshape(256, 2048)
    if arr.shape != (256, 2048):
        raise ValueError(f"latent has unexpected shape {arr.shape} for row {row_idx}")
    return arr.astype(np.float32, copy=False)


class RandomHistoryDataset(torch.utils.data.Dataset):
    """Wraps the normal train pipeline and only adds history latents for GRU.

    Uses base[idx] for observation + actions (same as train.py). Actions and
    observation are entirely produced by LeRobot + transforms; this class only
    adds the "latents" key for history frames.
    """

    def __init__(self, base_dataset, batch_size, window_size=5, *, action_horizon: int = 50):
        self.base = base_dataset
        self.window_size = window_size
        self.action_horizon = int(action_horizon)
        self.batch_size = batch_size
        self.half = batch_size // 2

        ds = base_dataset
        while hasattr(ds, "_dataset"):
            ds = ds._dataset
        self.hf_dataset = getattr(ds, "hf_dataset", None) or getattr(
            getattr(ds, "dataset", None), "hf_dataset", None
        )
        if self.hf_dataset is None:
            raise AttributeError(
                "RandomHistoryDataset: could not find underlying HuggingFace dataset. "
                f"Got base_dataset={type(base_dataset)}"
            )
        self.hf_raw = self.hf_dataset.with_format(None)

        # Latent-availability mask for history sampling only.
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

        is_decision = np.array(self.hf_raw["is_decision"])
        all_pos = np.where(is_decision == 1)[0]
        all_neg = np.where(is_decision == 0)[0]
        index_col = np.array(self.hf_raw["index"])
        self.pos_indices = np.array([i for i in all_pos if index_col[int(i)] >= self.window_size])
        self.neg_indices = np.array([i for i in all_neg if index_col[int(i)] >= self.window_size])
        logging.info(
            f"Initialized GRU Dataset: {len(self.pos_indices)} positive, {len(self.neg_indices)} negative samples."
        )

    def __len__(self):
        return len(self.base)

    def _history_indices(self, idx_int: int, is_decision: bool) -> np.ndarray:
        frame_row = self.hf_raw[idx_int]
        frame_in_ep = int(frame_row["index"])
        start_of_ep = idx_int - frame_in_ep
        past_indices = np.arange(start_of_ep, idx_int)
        sig_values = np.array(self.hf_raw[past_indices]["significance"])
        target_val = 1 if is_decision else 0
        valid_pool = past_indices[sig_values == target_val]
        if self._has_latent is not None and len(valid_pool):
            valid_pool = valid_pool[self._has_latent[valid_pool]]
        if len(valid_pool) == 0:
            return np.full(self.window_size, idx_int, dtype=np.int64)
        selected = np.random.choice(valid_pool, self.window_size, replace=True)
        selected.sort()
        return selected

    def _load_latents(self, indices: np.ndarray) -> np.ndarray:
        return np.stack([_load_latent_from_row(self.hf_raw, int(i)) for i in indices], axis=0)

    def _get_frame_with_history(self, idx: int, is_decision: bool) -> dict[str, typing.Any]:
        idx_int = int(idx)
        sample = self.base[idx_int]
        if not isinstance(sample, dict):
            sample = dict(sample) if hasattr(sample, "keys") else {"observation": sample, "actions": getattr(sample, "actions", None)}
        selected = self._history_indices(idx_int, is_decision)
        latents = self._load_latents(selected)
        return {**sample, "latents": latents}

    def __getitem__(self, batch_idx: int) -> dict[str, typing.Any]:
        slot_in_batch = batch_idx % self.batch_size
        if slot_in_batch < self.half:
            idx = int(np.random.choice(self.pos_indices))
            return self._get_frame_with_history(idx, is_decision=True)
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
    # LeRobot's default transform does torch.tensor(x) on every column; image columns
    # are stored as dicts (path/bytes) and raise "Could not infer dtype of dict".
    dataset.hf_dataset.set_transform(_safe_hf_transform_to_torch)

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
