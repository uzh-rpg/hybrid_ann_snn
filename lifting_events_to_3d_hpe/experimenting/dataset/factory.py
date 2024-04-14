"""
Factory module provide a set of Constructor for datasets using factory design
pattern.  It encapsulates core dataset implementation and implement
functionalities to get train, val, and test dataset
"""

from abc import ABC
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from ..utils import get_augmentation
from . import core
from .dataset import (
    AutoEncoderDataset,
    BaseDataset,
    ClassificationDataset,
    HeatmapDataset,
    Joints3DDataset,
    JointsDataset,
)

__all__ = [
    "BaseDataFactory",
    "ClassificationConstructor",
    "AutoEncoderConstructor",
    "Joints3DConstructor",
    "JointsConstructor",
    "HeatmapConstructor",
]


class BaseDataFactory(ABC):
    def __init__(self, dataset_task, core_dataset: core.BaseCore = None):
        self.dataset_task = dataset_task
        self.core_dataset = core_dataset

    def set_dataset_core(self, core_dataset: core.BaseCore):
        self.core_dataset = core_dataset

    def get_dataset(self, indexes, augmentation_config, **kwargs) -> Dataset:
        preprocess = get_augmentation(augmentation_config)
        return self.dataset_task(
            dataset=self.core_dataset, indexes=indexes, transform=preprocess, **kwargs
        )

    def get_frame_only_dataset(self, indexes, augmentation_config, **kwargs) -> Dataset:
        preprocess = get_augmentation(augmentation_config)
        return AutoEncoderDataset(
            dataset=self.core_dataset, indexes=indexes, transform=preprocess, **kwargs
        )

    def get_datasets(
        self, augmentation_train, augmentation_test, **kwargs
    ) -> Tuple[Dataset, Dataset, Dataset]:

        (train_indexes, val_indexes, test_indexes,) = self.get_train_test_split()
        preprocess_train = get_augmentation(augmentation_train)
        preprocess_val = get_augmentation(augmentation_test)

        return (
            self.dataset_task(
                dataset=self.core_dataset,
                indexes=train_indexes,
                transform=preprocess_train,
                **kwargs
            ),
            self.dataset_task(
                dataset=self.core_dataset,
                indexes=val_indexes,
                transform=preprocess_val,
                **kwargs
            ),
            self.dataset_task(
                dataset=self.core_dataset,
                indexes=test_indexes,
                transform=preprocess_val,
                **kwargs
            ),
        )

    def get_train_test_split(self, split_at=0.8):
        """
        Get train, val, and test indexes accordingly to partition function

        Args:
            split_at: Split train/val according to a given percentage
        Returns:
            Train, val, and test indexes as numpy vectors
        """
        data_indexes = np.arange(len(self.core_dataset.file_paths))
        test_indexes_mask = [
            self.core_dataset.partition_function(x) for x in data_indexes
        ]
        test_indexes = data_indexes[test_indexes_mask]

        train_indexes_mask = [
            self.core_dataset.train_partition_function(x) for x in data_indexes
        ]

        data_indexes = data_indexes[train_indexes_mask]

        train_indexes, val_indexes = _split_set(data_indexes, split_at=split_at)
        return train_indexes, val_indexes, test_indexes


class ClassificationConstructor(BaseDataFactory):
    def __init__(self):
        super(ClassificationConstructor, self).__init__(
            dataset_task=ClassificationDataset
        )


class JointsConstructor(BaseDataFactory):
    def __init__(self):
        super(JointsConstructor, self).__init__(dataset_task=JointsDataset)


class Joints3DConstructor(BaseDataFactory):
    def __init__(self):
        super(Joints3DConstructor, self).__init__(dataset_task=Joints3DDataset)


class Joints3DStereoConstructor(BaseDataFactory):
    def __init__(self):
        super(Joints3DStereoConstructor, self).__init__(dataset_task=Joints3DDataset)
        self.cams = [0, 1]

    def get_train_test_split(self, split_at=0.8):
        data_indexes = self.get_stereo_indexes()
        test_subject_indexes_mask = [
            self.core_dataset.partition_function(x)
            for x in self.core_dataset.file_paths[data_indexes[:, 0]]
        ]

        test_indexes = data_indexes[test_subject_indexes_mask, :]
        data_index = data_indexes[~np.in1d(data_indexes[:, 0], test_indexes[:, 0])]
        train_indexes, val_indexes = _split_set(data_index, split_at=split_at)

        return train_indexes, val_indexes, test_indexes

    def get_stereo_indexes(self):
        data_indexes = np.arange(len(self.core_dataset.file_paths))

        cam1_id = np.array(
            [
                idx
                for idx, file_path in zip(data_indexes, self.core_dataset.file_paths)
                if self.core_dataset.get_frame_info(file_path)['cam'] == self.cams[0]
            ]
        )

        cam2_id = np.array(
            [
                idx
                for idx, file_path in zip(data_indexes, self.core_dataset.file_paths)
                if self.core_dataset.get_frame_info(file_path)['cam'] == self.cams[1]
            ]
        )

        return np.stack([cam1_id, cam2_id], 1)


class HeatmapConstructor(BaseDataFactory):
    def __init__(self):
        super(HeatmapConstructor, self).__init__(dataset_task=HeatmapDataset)


class AutoEncoderConstructor(BaseDataFactory):
    def __init__(self):
        super(AutoEncoderConstructor, self).__init__(dataset_task=AutoEncoderDataset)


def _split_set(data_indexes, split_at=0.8):
    np.random.shuffle(data_indexes)
    n_data_for_training = len(data_indexes)
    train_split = int(split_at * n_data_for_training)
    train_indexes = data_indexes[:train_split]
    val_indexes = data_indexes[train_split:]

    return train_indexes, val_indexes
