"""
Base class for agents classes.  Each agent provides a dataset factory class to
get train, val, and test datasets.  Each agent must implement training,
validation, and test `steps` methods as well as `epoch_end` methods
"""

import os
import pdb
import hydra
import pytorch_lightning as pl
import torch

from ..dataset import BaseDataFactory
from ..utils import get_feature_extractor


class BaseModule(pl.LightningModule):
    def __init__(
        self, optimizer, lr_scheduler, loss, dataset_constructor, use_lr_scheduler
    ):
        """
        Base agent module
        """

        super(BaseModule, self).__init__()

        self.dataset_constructor = dataset_constructor
        self.optimizer_config = optimizer
        self.scheduler_config = lr_scheduler
        self.use_lr_scheduler = use_lr_scheduler

        self.loss_func = hydra.utils.instantiate(loss)

    def set_params(self):
        pass

    def get_data_factory(self) -> BaseDataFactory:
        return self.dataset_constructor()

    @staticmethod
    def _get_feature_extractor(model, n_channels, backbone_path, pretrained):
        extractor_params = {"n_channels": n_channels, "model": model}

        if backbone_path is not None and os.path.exists(backbone_path):
            extractor_params["custom_model_path"] = backbone_path
        else:
            if pretrained is not None:
                extractor_params["pretrained"] = pretrained
            else:
                extractor_params["pretrained"] = True

        feature_extractor = get_feature_extractor(extractor_params)

        return feature_extractor

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_config["type"])(
            params=self.parameters(), **self.optimizer_config["params"]
        )

        scheduler = None
        if self.use_lr_scheduler:
            scheduler = getattr(
                torch.optim.lr_scheduler, self.scheduler_config["type"]
            )(optimizer, **self.scheduler_config["params"])
            return [optimizer], [scheduler]

        return optimizer

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"avg_train_loss": avg_loss, "step": self.current_epoch}
        self.log_dict(logs)

    def _get_aggregated_results(self, outputs, prefix):
        results = {}

        for metric_key in self.metrics.keys():
            # mean along batch axis
            tensor_result = torch.stack([x[metric_key] for x in outputs]).mean()
            if len(tensor_result.shape) > 0:
                # list of values cannot be
                # converted to python numeric!
                tensor_result = tensor_result.tolist()
            else:
                tensor_result = float(tensor_result)

            results[prefix + metric_key] = tensor_result
        return results
