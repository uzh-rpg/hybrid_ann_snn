import glob
import json
import logging
import os
import shutil
import pdb
from pathlib import Path

import comet_ml
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import (
    CometLogger,
    LightningLoggerBase,
    TensorBoardLogger,
    WandbLogger,
)

import experimenting

from .utilities import get_checkpoint_path, instantiate_new_model, load_model

logging.basicConfig(level=logging.INFO)


class HydraTrainer:
    def __init__(self, cfg: DictConfig):
        self._trainer = pl.Trainer(**get_training_params(cfg))

        self.cfg = cfg
        # H36M 
        cfg.dataset['params']['data_dir'] = '/longtermdatasets/asude/lifting_dataset/ConstantCount'
        cfg.dataset['params']['joints_path'] = '/longtermdatasets/asude/lifting_dataset/ConstantCount/3d_joints.npz'
        # DHP19
        # cfg.dataset['params']['data_dir'] = '/scratch/asude/constant_count_frames/dhp19_scarpellini/frames'
        # cfg.dataset['params']['joints_path'] = '/scratch/asude/constant_count_frames/dhp19_scarpellini/labels'
        self.core = hydra.utils.instantiate(cfg.dataset)
        if ("load_path" in cfg and cfg['load_path'] is not None and os.path.exists(cfg['load_path'])):
            print("Loading training")
            self.model = load_model(
                cfg.load_path,
                cfg.training.module,
                model_zoo=cfg.model_zoo,
                core=self.core,
                test_metrics=['AUC', 'MPJPE', "PCK", 'MPJPE2D'],
                estimate_depth=cfg.training.estimate_depth
                # TODO should remove the following, as they're loaded from the checkpoint
                # backbone=cfg.training.backbone,
                # model=cfg.training.model,
            )
            self.exp_path = cfg['load_path']
        else:
            self.model = instantiate_new_model(cfg, self.core)
            self.exp_path = os.getcwd()

        self.data_module = experimenting.dataset.DataModule(core=self.core,num_workers=cfg.num_workers,batch_size=cfg.batch_size,dataset_factory=self.model.get_data_factory(),aug_train_config=cfg.augmentation_train,aug_test_config=cfg.augmentation_test,train_val_split=cfg.train_val_split if "train_val_split" in cfg else 0.8,)

    def get_raw_test_outputs(self):
        outputs = []
        for x in self.data_module.test_frame_only_dataloader():
            predictions, _ = self.model(x)
            outputs.append(predictions)

        return outputs

    def fit(self):
        """
        Launch training for a given config. Confs file can be found at /src/confs
        """
        try:
            self._trainer.fit(self.model, datamodule=self.data_module)
            return True
        except Exception as ex:
            _safe_train_end()
            print(ex)
            raise ex

    def test(self, save_results: bool = True):
        """
        Launch training for a given config. Confs file can be found at /src/confs

        Args:
           cfg (DictConfig): Config dictionary

        """
        # pdb.set_trace()
        results = self._trainer.test(self.model, datamodule=self.data_module)
        print(results)
        if save_results:
            result_path = os.path.join(self.exp_path, self.cfg.result_file)
            with open(result_path, 'w') as json_file:
                json.dump(results, json_file)

        return results


def get_training_params(cfg: DictConfig):
    """

    Parameters

    cfg: DictConfig :
        hydra configuration (examples in conf/train)

    -------

    """

    exp_path = os.getcwd()
    logger = [TensorBoardLogger(os.path.join(exp_path, "tb_logs"))]

    checkpoint_dir = os.path.join(exp_path, "checkpoints")
    log_profiler = os.path.join(exp_path, "profile.txt")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "{epoch:02d}-{val_loss:.2f}")
    )

    profiler = pl.profiler.AdvancedProfiler(log_profiler)
    gpus = cfg["gpus"]
    if type(gpus) == list or type(gpus) == ListConfig:
        gpus = [int(x) for x in gpus]

    trainer_configuration = {
        "gpus": gpus,
        "benchmark": True,
        "max_epochs": cfg["epochs"] if "epochs" in cfg else 100,
        "callbacks": [ckpt_cb],
        "track_grad_norm": 2,
        "weights_summary": "top",
        "logger": logger,
        "profiler": profiler,
    }

    if ((isinstance(gpus, list) and len(gpus) > 1)) or (
        (isinstance(gpus, int) and gpus > 1)
    ):
        if "accelerator" in cfg:
            trainer_configuration['accelerator'] = cfg["accelerator"]

    if "debug" in cfg and cfg['debug']:
        torch.autograd.set_detect_anomaly(debug)
        trainer_configuration["overfit_batches"] = 0.0005
        trainer_configuration["log_gpu_memory"] = True

    if "early_stopping" in cfg and cfg['early_stopping'] > 0:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=cfg["early_stopping"],
            verbose=False,
            mode="min",
        )
        trainer_configuration["callbacks"].append(early_stop_callback)

    if "resume" in cfg and cfg["resume"]:
        checkpoint_path = get_checkpoint_path(cfg['load_path'])
        trainer_configuration["resume_from_checkpoint"] = checkpoint_path

    return trainer_configuration


def _safe_train_end():
    exp_path = Path(os.getcwd())
    exp_name = exp_path.name
    error_path = os.path.join(exp_path.parent, "with_errors", exp_name)
    shutil.move(exp_path, error_path)
    logging.log(logging.ERROR, "exp moved to trash")


def _get_comet_logger(exp_name: str, project_name: str) -> LightningLoggerBase:
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        experiment_name=exp_name,
        project_name=project_name,
    )
    return comet_logger


def _get_wandb_logger(exp_name: str, project_name: str) -> LightningLoggerBase:
    logger = WandbLogger(name=exp_name, project=project_name,)
    return logger
