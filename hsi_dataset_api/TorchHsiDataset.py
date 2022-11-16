from torch.utils.data import IterableDataset
import numpy as np

from hsi_dataset_api import HsiDataset


class TorchDatasetWrapper(IterableDataset):
    def __init__(self, dataset, transforms=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        
    def __iter__(self):
        return self.create_iterator()
    
    def create_iterator(self):
        for datapoint in self.dataset.data_iterator(opened=True):
            hsi = datapoint.hsi
            mask = datapoint.mask[..., 0]
            if self.transforms is not None:
                yield self.transforms(hsi), mask
            else:
                yield hsi, mask
