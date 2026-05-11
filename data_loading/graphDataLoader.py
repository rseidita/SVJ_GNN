from utils.JetGraphProducer import JetGraphProducer
from torch_geometric.loader import DataLoader
from typing import Optional, List
from utils.LundTreeUtilities import OnTheFlyNormalizer
from torch.utils.data import ConcatDataset, Sampler
import numpy as np
import os
import torch
import warnings

class ChunkSortedWeightedSampler(Sampler):
    """Weighted random sampler that re-orders the drawn indices by on-disk
    chunk so that each chunk file is opened once and fully consumed before
    the next one is touched.  Statistically equivalent to
    ``WeightedRandomSampler`` (same probability of drawing each sample) but
    minimises redundant chunk-file I/O.

    Args:
        weights (Tensor): A 1-D tensor of per-sample sampling weights.
        num_samples (int): Number of indices to draw per epoch.
        datasets (list): The list of ``JetGraphProducer`` datasets whose
            ``chunk_size`` attributes are used to compute sort keys.
        replacement (bool): Whether to sample with replacement (default:
            ``True``).

    Note:
        Within each epoch the sampler yields all samples from chunk 0 before
        chunk 1, etc.  If mini-batch diversity matters, make sure the
        DataLoader's ``batch_size`` is smaller than a single chunk.
    """

    def __init__(self, weights, num_samples, datasets, replacement=True):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

        # Build a sort-key array: assign every position in the concatenated
        # dataset an integer "chunk rank" so that all positions sharing the
        # same on-disk chunk file get the same rank and ranks increase
        # monotonically across chunks (and datasets).
        sort_keys = []
        chunk_rank = 0
        for ds in datasets:
            cs = getattr(ds, 'chunk_size', len(ds))
            # ds.indices() returns range(len(ds)) for an unsliced dataset or
            # a tensor of storage-level indices for a sliced/filtered one.
            raw = ds.indices()
            if isinstance(raw, torch.Tensor):
                storage_indices = raw.to(dtype=torch.long)
            else:
                storage_indices = torch.arange(len(ds), dtype=torch.long)
            local_chunk_ids = storage_indices // cs
            sort_keys.append(local_chunk_ids + chunk_rank)
            chunk_rank += int(local_chunk_ids.max().item()) + 1 if len(local_chunk_ids) else 0
        self._sort_keys = torch.cat(sort_keys)

    def __iter__(self):
        # 1. Draw all indices for this epoch using the weighted scheme.
        indices = torch.multinomial(self.weights, self.num_samples,
                                    replacement=self.replacement)
        # 2. Sort by chunk rank so the chunk cache is maximally effective.
        order = torch.argsort(self._sort_keys[indices], stable=True)
        return iter(indices[order].tolist())

    def __len__(self):
        return self.num_samples


class GraphDataLoader(DataLoader):
    """
    A custom DataLoader for loading JetGraphProducer objects. It allows for the
    use of a normalizer to be applied to the data, and for the specification of
    a window of the data to be loaded. It also allows for the loading of multiple
    datasets at once, and concatenates them into a single dataset.

    Args:
        dataset_paths (List[List]): A list of lists, where each inner list contains
            the path to the dataset and the name of the dataset to be loaded.
        normalizer (OnTheFlyNormalizer): A normalizer to be applied to the data.
            It is registered as a per-graph ``transform`` on each dataset so that
            normalisation happens lazily at access time rather than up-front.
        window_to_load (List[List[float]]): A list of lists, where each inner list
            contains the start and end of the window to be loaded. If None, the
            entire dataset is loaded.
        batch_size (int): The batch size to be used. Negative numbers will read as len(dataset)
        dataset_fraction (List[float]): List of weights in order to have balanced datasets.
        shuffle (bool): Whether to shuffle the data.
        follow_batch (Optional[List[str]]): A list of strings specifying which
            attributes of the data object should be batched together. If None,
            the default behavior is used.
        exclude_keys (Optional[List[str]]): A list of strings specifying which
            attributes of the data object should not be batched together. If None,
            the default behavior is used.
        device (torch.device): Kept for API compatibility.  With the chunk-based
            dataset, moving data to a specific device should be done in the
            training loop (e.g. ``batch.to(device)``) rather than pre-loading
            everything onto the device.
        jet_idx (Optional(int)): An integer to filter for the corresponding jet (pt ordered) in every event
        sample (Optional(Bool)): Wether or not to use weighted random sampling (True by default)
        **kwargs: Additional keyword arguments to be passed to the DataLoader.

    Returns:
        DataLoader: A DataLoader object for loading JetGraphProducer objects.
    """
    def __init__(
            self,
            dataset_paths: List[List],
            normalizer: OnTheFlyNormalizer=None,
            window_to_load: List[List[float]]=None,
            batch_size: int=1,
            dataset_fraction: List[float]=None,
            shuffle: bool = False,
            permutation: Optional[dict] = None,
            follow_batch: Optional[List[str]] = None,
            exclude_keys: Optional[List[str]] = None,
            device: torch.device=None,
            jet_idx: int = None,
            sample = True,
            **kwargs,
    ):
        self.dataset_paths = dataset_paths
        self.normalizer = normalizer
        self.window_to_load = window_to_load
        self.shuffle = shuffle
        self.permutation = permutation if permutation is not None else {}
        if self.window_to_load is None:
            self.window_to_load = [[0., 1.] for _ in dataset_paths]
        self.device = device
        self.jet_idx = jet_idx

        if device is not None:
            warnings.warn(
                "The 'device' parameter is deprecated. With the chunk-based dataset, "
                "move batches to the target device in your training loop "
                "(e.g. `batch = batch.to(device)`) instead of pre-loading everything "
                "onto the device. The parameter is kept for backwards compatibility but "
                "has no effect on the dataset itself.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Consistency checks
        assert len(self.window_to_load) == len(self.dataset_paths), "Please provide one slicing window per dataset!"
        assert (not self.permutation) or (len(self.permutation) == len(self.dataset_paths)), \
            "Mismatch between number of datasets and permutations, please check for consistency"
        
        
        datasets = []
        weights = []
        # Build the dataset
        for i, (path, name) in enumerate(dataset_paths):
            meta_file = f"{path}/processed/{name}_meta.pt"
            if not os.path.exists(meta_file):
                raise FileNotFoundError(
                    f"The requested dataset {path}/{name} has not been processed, "\
                    "please do so before attempting to load it."
                        )
            # Pass the normalizer as a per-graph transform so it is applied
            # lazily at access time instead of requiring all data in RAM.
            data = JetGraphProducer(path, output_dataset_name=name, transform=normalizer)
            
            if self.shuffle:
                if name not in self.permutation.keys():
                    self.permutation[name] = torch.randperm(len(data))
                data = data[self.permutation[name]]
            
            data = data[int(self.window_to_load[i][0]*len(data)):int(self.window_to_load[i][1]*len(data))]
            weights.append(data.w)
            
            if self.jet_idx is not None:
                data = data[data.jet_idx == self.jet_idx]

            if dataset_fraction is not None:
                weights[i] = weights[i]*dataset_fraction[i]/np.sum(dataset_fraction)/weights[i].sum()

            datasets.append(data)

        data = ConcatDataset(datasets)
        weights = torch.cat(weights).to(device) if device is not None else torch.cat(weights)
        sampler = ChunkSortedWeightedSampler(weights, len(data), datasets) if sample else None
        batch_size = batch_size if batch_size > 0 else len(data)

        super(GraphDataLoader, self).__init__(data, batch_size, False, follow_batch, exclude_keys, sampler=sampler, **kwargs)

    def get_permutation(self):
        return self.permutation
