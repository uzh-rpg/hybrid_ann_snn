"""
Toolbox for DHP19 evaluation procedure
"""
import collections
import json
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig

import experimenting

from ..dataset import Joints3DConstructor
from ..dataset.datamodule import get_dataloader
from .trainer import HydraTrainer
from .utilities import get_checkpoint_path, load_model


class PerMovementEvaluator(HydraTrainer):
    def __init__(self, cfg, n_movements=33):
        super().__init__(cfg)
        self.cfg = cfg
        self.n_movements = n_movements

    def _get_test_loaders_iterator(self):
        factory = Joints3DConstructor()

        for movement in range(0, self.n_movements):
            self.cfg.dataset.params.movements = [
                movement
            ]  # test on 1 movement at the time
            core_per_movement = hydra.utils.instantiate(self.cfg.dataset)
            factory.set_dataset_core(core_per_movement)
            _, _, test = factory.get_datasets(
                self.cfg.augmentation_train, self.cfg.augmentation_test
            )

            loader = get_dataloader(
                dataset=test, batch_size=32, shuffle=False, num_workers=12
            )
            yield loader

    def evaluate_per_movement(self):
        """
        Retrieve trained agent using cfg and apply its evaluation protocol to
        extract results

        Args: cfg (omegaconf.DictConfig): Config dictionary (need to specify a
              load_path and a training task)


        Returns:
            Results obtained applying the dataset evaluation protocol, per metric
        """

        test_metrics = self.cfg.training.test_metrics
        metrics = ["test_mean" + k for k in test_metrics]

        final_results = collections.defaultdict(dict)
        test_loaders = self._get_test_loaders_iterator()
        trainer = pl.Trainer(gpus=self.cfg.gpus)

        for loader_id, loader in enumerate(test_loaders):
            results = trainer.test(self.model, test_dataloaders=loader)[0]

            print(f"Step {loader_id}")
            print(results)
            for metric in metrics:
                tensor_result = results[metric]
                final_results[metric][f"movement_{loader_id}"] = tensor_result

        return final_results
