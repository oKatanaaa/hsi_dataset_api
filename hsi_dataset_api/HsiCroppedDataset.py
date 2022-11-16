from glob import glob
import os
from dataclasses import dataclass
from typing import List, Tuple, Union
import warnings

import cv2
import numpy as np
import yaml

from .utils import WrongDirectoryStructure
from .Dataset import HsiDatapoint


@dataclass
class HsiCropset:
    datapoints: List[HsiDatapoint]
    meta: dict


class HsiCroppedDataset:
    def __init__(self, path_to_dataset: str = None, cropset_points: List[HsiCropset] = None):
        if path_to_dataset is not None:
            self._load_from_path(path_to_dataset)
        elif cropset_points is not None:
            self.cropset_points = cropset_points
            self.dataset_description = None
        else:
            raise RuntimeError('Either of the arguments must be supplied (path_to_dataset or cropset_points).')
    
    def _load_from_path(self, path_to_dataset):
        self._check_folder_structure(path_to_dataset)
        self.path_to_dataset = path_to_dataset

        self.dataset_description = self._read_dataset_description()
        folders = glob(os.path.join(self.path_to_dataset, 'hsi', '*'))
        folders = map(lambda x: x.split(os.path.sep)[-1], folders)
        self.cropset_points = [self.read_cropset_metaonly(folder) for folder in folders]
    
    def read_cropset_metaonly(self, folder: str) -> HsiCropset:
        hsi_filenames = sorted(glob(os.path.join(self.path_to_dataset, 'hsi', folder, '*.npy')))
        meta_hsi_filenames = sorted(glob(os.path.join(self.path_to_dataset, 'hsi', folder, '*.yml')))
        mask_filenames = sorted(glob(os.path.join(self.path_to_dataset, 'masks', folder, '*.png')))
        
        cropset = HsiCropset([], {'hsi_id': folder, 'classes': set()})
        for hsi, meta, mask in zip(hsi_filenames, meta_hsi_filenames, mask_filenames):
            datapoint = HsiDatapoint(
                hsi=hsi,
                mask=mask,
                meta=self._read_yml(meta)
            )
            cropset.datapoints.append(datapoint)
            [cropset.meta['classes'].add(_cls) for _cls in datapoint.meta['classes']]
        return cropset

    def _check_folder_structure(self, path_to_dataset: str):
        """

        :param path_to_dataset:
        :return:
        """
        msg = 'Wrong dataset directory structure.'

        required_files = {'hsi', 'masks', 'meta.yml'}
        if len(set(os.listdir(path_to_dataset)) - required_files) != 0:
            raise WrongDirectoryStructure(msg + f"In {path_to_dataset}")

    def _read_yml(self, path):
        with open(path, 'r') as f:
            return yaml.full_load(f)

    def _read_dataset_description(self):
        return self._read_yml(os.path.join(self.path_to_dataset, 'meta.yml'))

    def get_dataset_description(self) -> dict:
        return self.dataset_description
    
    def data_iterator(self, opened: bool = True):
        """
        RAM friendly method that returns a generator of HSI data points. Shuffles data by default.
        """
        choose_array = np.random.permutation(len(self.cropset_points))
        for idx in choose_array:
            cropset = self.cropset_points[idx]

            _idx = np.random.randint(low=0, high=len(cropset.datapoints))
            dp = cropset.datapoints[_idx]
            if opened:
                hsi_datapoint = HsiDatapoint(
                    hsi=np.load(dp.hsi),
                    mask=cv2.imread(dp.mask),
                    meta=dp.meta
                )
            else:
                hsi_datapoint = dp
            yield hsi_datapoint

    def data_readall_as_list(self, opened: bool = True, shuffle: bool = False) \
            -> List[HsiDatapoint]:
        """
        Returns list of objects, RAM consuming if opened flag set TRUE

        :param shuffle: if TRUE returns data in shuffled way
        :param opened: if TRUE returns opened data points i.e np.ndarrays instead paths to the data
        :return: list with HsiDatapoints
        """
        return list(self.data_iterator(opened, shuffle))
    
    def select_traintest_set(self, seed=1):
        classes = set()
        for cropset in self.cropset_points:
            classes.update(cropset.meta['classes'])
        
        cls2cropsets = {}
        for _cls in classes:
            cropsets = []
            for cropset in self.cropset_points:
                if _cls in cropset.meta['classes']:
                    cropsets.append(cropset)
            cls2cropsets[_cls] = cropsets
        
        test_set = []
        test_classes = set()
        used_hsi = set()
        gen = np.random.RandomState(seed=seed)
        for _cls, cropsets in cls2cropsets.items():
            iters = 0
            while True:
                idx = gen.randint(low=0, high=len(cropsets))
                cropset = cropsets[idx]
                iters += 1
                if not (cropset.meta['hsi_id'] in used_hsi):
                    used_hsi.add(cropset.meta['hsi_id'])
                    test_classes.update(cropset.meta['classes'])
                    break
                
                if iters > 1000:
                    warnings.warn(
                        f'Could not find a free sample containing class={_cls}. '
                        'Test set will be incomplete. Try using different seed.')
                    break
                    
            test_set.append(cropset)
            used_hsi.add(cropset.meta['hsi_id'])
            if test_classes == classes:
                break
        
        train_set = []
        for cropset in self.cropset_points:
            if cropset.meta['hsi_id'] in used_hsi:
                continue
            train_set.append(cropset)
        return HsiCroppedDataset(cropset_points=train_set), HsiCroppedDataset(cropset_points=test_set)