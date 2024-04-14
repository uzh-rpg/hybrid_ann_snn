import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pdb
from .core import BaseCore
from .factory import BaseDataFactory


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_factory: BaseDataFactory,
        core: BaseCore,
        aug_train_config,
        aug_test_config,
        batch_size: int,
        num_workers: int,
        train_val_split: float = 0.8,
    ):
        super().__init__()
        self.core = core
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_factory = dataset_factory

        self.aug_train_config = aug_train_config
        self.aug_test_config = aug_test_config
        self.train_val_split = train_val_split

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        self.dataset_factory.set_dataset_core(self.core)
        (
            self.train_indexes,
            self.val_indexes,
            self.test_indexes,
        ) = self.dataset_factory.get_train_test_split(self.train_val_split)

        self.train_dataset = self.dataset_factory.get_dataset(
            self.train_indexes, self.aug_train_config
        )
        self.val_dataset = self.dataset_factory.get_dataset(
            self.val_indexes, self.aug_test_config
        )
        self.test_dataset = self.dataset_factory.get_dataset(
            self.test_indexes, self.aug_test_config
        )

    def train_dataloader(self):
        return get_dataloader(self.train_dataset, self.batch_size, self.num_workers)

    def val_dataloader(self):
        return get_dataloader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return get_dataloader(
            self.test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_frames_only_dataloader(self):
        return get_dataloader(
            self.dataset_factory.get_frame_only_dataset(
                self.test_indexes, self.aug_test_config
            ),
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int = 12, shuffle=True
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
