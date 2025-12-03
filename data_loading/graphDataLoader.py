from utils.LundGraphProducer import LundGraphProducer
from torch_geometric.loader import DataLoader
from typing import Optional, List
from utils.LundTreeUtilities import OnTheFlyNormalizer
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import numpy as np
import os
import torch

class GraphDataLoader(DataLoader):
    """
    A custom DataLoader for loading LundGraphProducer objects. It allows for the
    use of a normalizer to be applied to the data, and for the specification of
    a window of the data to be loaded. It also allows for the loading of multiple
    datasets at once, and concatenates them into a single dataset.

    Args:
        dataset_paths (List[List]): A list of lists, where each inner list contains
            the path to the dataset and the name of the dataset to be loaded. Optionally,
            a third element can be provided to specify the processed dataset path.
        normalizer (OnTheFlyNormalizer): A normalizer to be applied to the data.
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
        jet_idx (Optional(int)): An integer to filter for the corresponding jet (pt ordered) in every event
        sample (Optional(Bool)): Wether or not to use weighted random sampling (True by default)
        **kwargs: Additional keyword arguments to be passed to the DataLoader.

    Returns:
        DataLoader: A DataLoader object for loading LundGraphProducer objects.
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

        # Consistency checks
        assert len(self.window_to_load) == len(self.dataset_paths), "Please provide one slicing window per dataset!"
        assert (not self.permutation) or (len(self.permutation) == len(self.dataset_paths)), \
            "Mismatch between number of datasets and permutations, please check for consistency"
        
        
        datasets = []
        weights = []
        # Build the dataset
        for i, dataset_info in enumerate(dataset_paths):
            path = dataset_info[0]
            name = dataset_info[1]
            processed_path = dataset_info[2] if len(dataset_info) > 2 else None
            
            # Determine the actual processed file path
            if processed_path is not None:
                expected_file = os.path.join(processed_path, f"{name}.pt")
            else:
                expected_file = os.path.join(path, "processed", f"{name}.pt")
            
            if not os.path.exists(expected_file):
                raise FileNotFoundError(
                    f"The requested dataset {expected_file} has not been processed, "\
                    "please do so before attempting to load it."
                        )
            data = LundGraphProducer(path, processed_dataset_name=name, processed_dataset_path=processed_path)
            
            if self.device is not None:
                data.data.to(self.device)
            
            if self.shuffle:
                if name not in self.permutation.keys():
                    self.permutation[name] = torch.randperm(len(data))
                data = data[self.permutation[name]]
            
            data = data[int(self.window_to_load[i][0]*len(data)):int(self.window_to_load[i][1]*len(data))]
            # FIXME: reintroduce when weights are back
            # weights.append(data.w)
            weights.append(torch.ones(len(data)))
            
            if self.jet_idx is not None:
                data = data[data.jet_idx == self.jet_idx]

            if dataset_fraction is not None:
                weights[i] = weights[i]*dataset_fraction[i]/np.sum(dataset_fraction)/weights[i].sum()
            
            if normalizer is not None:
                normalizer(data.data)

            datasets.append(data)

        data = ConcatDataset(datasets)
        weights = torch.cat(weights).to(device) if device is not None else torch.cat(weights)
        sampler = WeightedRandomSampler(weights, len(data), replacement=True) if sample else None
        batch_size = batch_size if batch_size > 0 else len(data)

        super(GraphDataLoader, self).__init__(data, batch_size, False, follow_batch, exclude_keys, sampler=sampler, **kwargs)

    def get_permutation(self):
        return self.permutation
