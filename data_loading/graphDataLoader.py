import hashlib
import warnings
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import ConcatDataset, Sampler
from torch_geometric.loader import DataLoader

from utils.JetGraphProducer import JetGraphProducer
from utils.LundTreeUtilities import OnTheFlyNormalizer


_SPLIT_STATE_VERSION = 1
_SAMPLING_MODES = {"sequential", "shuffle", "weighted"}


class ChunkAwareWeightedSampler(Sampler):
    """Draw weighted samples, then order the draw by randomized chunk.

    Grouping happens after ``torch.multinomial``. The sampled multiset,
    including repetitions, is therefore identical to ordinary weighted
    sampling; only its traversal order changes.
    """

    def __init__(self, weights, chunk_ids, num_samples, generator=None):
        self.weights = torch.as_tensor(
            weights, dtype=torch.double
        ).detach().cpu()
        self.chunk_ids = torch.as_tensor(
            chunk_ids, dtype=torch.long
        ).detach().cpu()
        self.num_samples = int(num_samples)
        self.generator = generator

        if self.weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if self.chunk_ids.shape != self.weights.shape:
            raise ValueError("chunk_ids and weights must have the same shape")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")

    def __iter__(self):
        sampled = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=self.generator,
        )
        sampled_chunks = self.chunk_ids[sampled]
        _, inverse_chunks = torch.unique(
            sampled_chunks, return_inverse=True
        )
        n_chunks = int(inverse_chunks.max()) + 1
        chunk_order = torch.randperm(n_chunks, generator=self.generator)
        chunk_rank = torch.empty(n_chunks, dtype=torch.long)
        chunk_rank[chunk_order] = torch.arange(n_chunks)

        # A stable sort preserves the multinomial ordering within each chunk.
        # Assigning randomized ranks first randomizes the order of chunks.
        traversal_order = torch.argsort(
            chunk_rank[inverse_chunks], stable=True
        )
        yield from sampled[traversal_order].tolist()

    def __len__(self):
        return self.num_samples


def _dataset_key(path, name):
    return f"{Path(path).resolve()}::{name}"


def _file_sha256(file_name):
    digest = hashlib.sha256()
    with open(file_name, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_window(window, dataset_key):
    if len(window) != 2:
        raise ValueError(
            f"Split window for {dataset_key!r} must contain [start, stop]"
        )
    start, stop = float(window[0]), float(window[1])
    if not (0.0 <= start <= stop <= 1.0):
        raise ValueError(
            f"Invalid split window {window!r} for {dataset_key!r}; "
            "expected 0 <= start <= stop <= 1"
        )
    return start, stop


def _validate_permutation(permutation, length, dataset_key):
    permutation = torch.as_tensor(permutation, dtype=torch.long).cpu()
    if permutation.ndim != 1 or len(permutation) != length:
        raise ValueError(
            f"Split permutation for {dataset_key!r} has length "
            f"{len(permutation)}, expected {length}"
        )
    if length and (
        permutation.min().item() != 0
        or permutation.max().item() != length - 1
        or torch.unique(permutation).numel() != length
    ):
        raise ValueError(
            f"Split permutation for {dataset_key!r} is not a permutation "
            f"of [0, {length})"
        )
    return permutation


class GraphDataLoader(DataLoader):
    """Load, partition, combine, and sample processed graph datasets.

    Dataset partitioning and per-epoch sampling are deliberately independent:

    1. A fixed permutation from ``split_state`` or ``split_seed`` is applied.
    2. ``split_windows`` selects a stable partition of each dataset.
    3. Optional jet-index filtering is applied.
    4. ``sampling`` controls ordering within the selected partition.

    Args:
        dataset_paths: ``[(root, processed_dataset_name), ...]``.
        normalizer: Optional lazy per-graph transform.
        split_windows: One fractional ``(start, stop)`` window per dataset.
        split_seed: Generate deterministic randomized splits when not ``None``.
            ``None`` preserves original dataset order.
        split_state: Previously returned state used to reproduce exact splits.
        sampling: ``"sequential"``, ``"shuffle"``, or ``"weighted"``.
        sampling_weights: For weighted sampling, ``"event"`` uses saved graph
            weights and ``"uniform"`` assigns equal weight to every graph.
        dataset_fraction: Optional target sampling mass for each dataset.
        generator: Optional PyTorch generator for epoch shuffling/sampling.
        jet_idx: Optionally retain only a particular pT-ordered jet.

    ``window_to_load``, ``permutation``, ``shuffle``, and ``sample`` are
    deprecated compatibility aliases.
    """

    def __init__(
        self,
        dataset_paths: List[List],
        normalizer: OnTheFlyNormalizer = None,
        split_windows: Optional[List[List[float]]] = None,
        split_seed: Optional[int] = None,
        split_state: Optional[dict] = None,
        sampling: Optional[str] = None,
        sampling_weights: str = "event",
        batch_size: int = 1,
        dataset_fraction: Optional[List[float]] = None,
        generator: Optional[torch.Generator] = None,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        jet_idx: Optional[int] = None,
        device: torch.device = None,
        # Deprecated aliases:
        window_to_load: Optional[List[List[float]]] = None,
        permutation: Optional[dict] = None,
        shuffle: Optional[bool] = None,
        sample: Optional[bool] = None,
        **kwargs,
    ):
        if not dataset_paths:
            raise ValueError("dataset_paths must contain at least one dataset")

        if split_windows is not None and window_to_load is not None:
            raise ValueError(
                "Use split_windows or deprecated window_to_load, not both"
            )
        if split_windows is None:
            split_windows = window_to_load
            if window_to_load is not None:
                warnings.warn(
                    "'window_to_load' is deprecated; use split_windows",
                    DeprecationWarning,
                    stacklevel=2,
                )
        if split_windows is None:
            split_windows = [[0.0, 1.0] for _ in dataset_paths]
        if len(split_windows) != len(dataset_paths):
            raise ValueError("Provide one split window per dataset")

        if split_state is not None and permutation:
            raise ValueError(
                "Use split_state or deprecated permutation, not both"
            )
        if permutation:
            warnings.warn(
                "'permutation' is deprecated; use split_state",
                DeprecationWarning,
                stacklevel=2,
            )
        if split_state is not None and split_seed is not None:
            raise ValueError("split_state and split_seed are mutually exclusive")
        if permutation and split_seed is not None:
            raise ValueError(
                "deprecated permutation and split_seed are mutually exclusive"
            )
        if shuffle is not None:
            warnings.warn(
                "'shuffle' is deprecated for split construction; use "
                "split_seed for a fixed randomized split and sampling='shuffle' "
                "for per-epoch shuffling",
                DeprecationWarning,
                stacklevel=2,
            )
            if shuffle and split_seed is None and split_state is None:
                split_seed = int(torch.initial_seed())

        if sampling is None:
            if sample is not None:
                warnings.warn(
                    "'sample' is deprecated; use sampling='weighted' or "
                    "sampling='sequential'",
                    DeprecationWarning,
                    stacklevel=2,
                )
                sampling = "weighted" if sample else "sequential"
            else:
                sampling = "sequential"
        elif sample is not None:
            legacy_sampling = "weighted" if sample else "sequential"
            if sampling != legacy_sampling:
                raise ValueError("sampling and deprecated sample disagree")
        if sampling not in _SAMPLING_MODES:
            raise ValueError(
                f"sampling must be one of {sorted(_SAMPLING_MODES)}, "
                f"got {sampling!r}"
            )
        if sampling_weights not in {"event", "uniform"}:
            raise ValueError("sampling_weights must be 'event' or 'uniform'")

        if dataset_fraction is not None:
            if sampling != "weighted":
                raise ValueError(
                    "dataset_fraction is only meaningful with "
                    "sampling='weighted'"
                )
            if len(dataset_fraction) != len(dataset_paths):
                raise ValueError(
                    "dataset_fraction must contain one value per dataset"
                )
            fractions = torch.as_tensor(dataset_fraction, dtype=torch.double)
            if (
                not torch.isfinite(fractions).all()
                or (fractions < 0).any()
                or fractions.sum() <= 0
            ):
                raise ValueError(
                    "dataset_fraction values must be finite, nonnegative, "
                    "and have a positive sum"
                )
            fractions = fractions / fractions.sum()
        else:
            fractions = None

        if device is not None:
            warnings.warn(
                "The 'device' parameter is deprecated and has no effect. Move "
                "batches to the target device in the training loop.",
                DeprecationWarning,
                stacklevel=2,
            )

        if split_state is not None:
            if split_state.get("version") != _SPLIT_STATE_VERSION:
                raise ValueError(
                    f"Unsupported split state version "
                    f"{split_state.get('version')!r}"
                )
            state_datasets = split_state.get("datasets")
            if not isinstance(state_datasets, dict):
                raise ValueError("split_state has no valid datasets mapping")
        else:
            state_datasets = {}

        datasets = []
        weights = []
        chunk_ids = []
        chunk_offset = 0
        generated_state = {
            "version": _SPLIT_STATE_VERSION,
            "seed": (
                split_state.get("seed")
                if split_state is not None
                else split_seed
            ),
            "datasets": {},
        }
        compatibility_permutations = {}

        for index, ((path, name), window) in enumerate(
            zip(dataset_paths, split_windows)
        ):
            key = _dataset_key(path, name)
            metadata_file = Path(path) / "processed" / f"{name}_meta.pt"
            if not metadata_file.exists():
                raise FileNotFoundError(
                    f"Processed dataset metadata not found: {metadata_file}"
                )

            data = JetGraphProducer(
                path,
                output_dataset_name=name,
                transform=normalizer,
                validate_cache_config=False,
            )
            length = len(data)
            fingerprint = _file_sha256(metadata_file)

            if split_state is not None:
                if key not in state_datasets:
                    raise ValueError(
                        f"split_state has no entry for dataset {key!r}"
                    )
                record = state_datasets[key]
                if record.get("length") != length:
                    raise ValueError(
                        f"Dataset length changed for {key!r}: split state has "
                        f"{record.get('length')}, dataset has {length}"
                    )
                if record.get("metadata_sha256") != fingerprint:
                    raise ValueError(
                        f"Processed metadata changed for {key!r}; regenerate "
                        "the split state"
                    )
                dataset_permutation = _validate_permutation(
                    record.get("permutation"), length, key
                )
            elif permutation:
                legacy_key = key if key in permutation else name
                if legacy_key not in permutation:
                    raise ValueError(
                        f"Deprecated permutation has no entry for {key!r}"
                    )
                dataset_permutation = _validate_permutation(
                    permutation[legacy_key], length, key
                )
            elif split_seed is not None:
                # Derive a path-stable per-dataset seed so dataset ordering does
                # not change any existing split.
                seed_material = f"{split_seed}:{key}".encode()
                dataset_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:8], "little"
                ) % (2**63 - 1)
                split_generator = torch.Generator().manual_seed(dataset_seed)
                dataset_permutation = torch.randperm(
                    length, generator=split_generator
                )
            else:
                dataset_permutation = torch.arange(length)

            generated_state["datasets"][key] = {
                "length": length,
                "metadata_sha256": fingerprint,
                "permutation": dataset_permutation.clone(),
            }
            compatibility_permutations[key] = dataset_permutation.clone()

            start_fraction, stop_fraction = _validate_window(window, key)
            start = int(start_fraction * length)
            stop = int(stop_fraction * length)
            selected_indices = dataset_permutation[start:stop]
            # The permutation determines partition membership, not physical
            # traversal order. Sorting the selected global indices lets
            # sequential loaders stream each graph chunk once instead of
            # repeatedly evicting and reloading chunks in permutation order.
            selected_indices = selected_indices.sort().values
            data = data[selected_indices]

            if jet_idx is not None:
                data = data[data.jet_idx == jet_idx]
            if len(data) == 0:
                raise ValueError(
                    f"Dataset partition for {key!r} is empty after windowing "
                    "and filtering"
                )

            if sampling == "weighted":
                original_indices = torch.as_tensor(
                    data.indices(), dtype=torch.long
                )
                chunk_ids.append(
                    original_indices // data.graphs_per_chunk + chunk_offset
                )
                chunk_offset += (
                    length + data.graphs_per_chunk - 1
                ) // data.graphs_per_chunk

                if sampling_weights == "event":
                    data_weights = data.w.detach().cpu().to(torch.double)
                else:
                    data_weights = torch.ones(len(data), dtype=torch.double)
                if (
                    not torch.isfinite(data_weights).all()
                    or (data_weights < 0).any()
                ):
                    raise ValueError(
                        f"Sampling weights for {key!r} must be finite and "
                        "nonnegative"
                    )
                if fractions is not None:
                    total_weight = data_weights.sum()
                    if total_weight <= 0:
                        raise ValueError(
                            f"Sampling weights for {key!r} have zero total "
                            "weight"
                        )
                    data_weights = (
                        data_weights / total_weight * fractions[index]
                    )
                weights.append(data_weights)

            datasets.append(data)

        combined_dataset = ConcatDataset(datasets)
        combined_weights = None
        if sampling == "weighted":
            combined_weights = torch.cat(weights)
            if combined_weights.sum() <= 0:
                raise ValueError(
                    "Weighted sampling requires positive total weight"
                )

        sampler = None
        epoch_shuffle = sampling == "shuffle"
        if sampling == "weighted":
            sampler = ChunkAwareWeightedSampler(
                combined_weights,
                torch.cat(chunk_ids),
                num_samples=len(combined_dataset),
                generator=generator,
            )

        if batch_size <= 0:
            batch_size = len(combined_dataset)

        self.dataset_paths = dataset_paths
        self.normalizer = normalizer
        self.split_windows = split_windows
        self.window_to_load = split_windows
        self.split_seed = split_seed
        self.split_state = generated_state
        self.permutation = compatibility_permutations
        self.sampling = sampling
        self.jet_idx = jet_idx

        super().__init__(
            dataset=combined_dataset,
            batch_size=batch_size,
            shuffle=epoch_shuffle,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
            sampler=sampler,
            generator=generator,
            **kwargs,
        )

    def get_split_state(self):
        """Return a cloneable, serializable description of fixed partitions."""
        return {
            "version": self.split_state["version"],
            "seed": self.split_state["seed"],
            "datasets": {
                key: {
                    "length": record["length"],
                    "metadata_sha256": record["metadata_sha256"],
                    "permutation": record["permutation"].clone(),
                }
                for key, record in self.split_state["datasets"].items()
            },
        }

    def get_permutation(self):
        """Deprecated compatibility accessor; prefer :meth:`get_split_state`."""
        return {
            key: value.clone() for key, value in self.permutation.items()
        }
