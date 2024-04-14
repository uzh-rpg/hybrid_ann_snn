import glob
import os
import pdb 

import pytorch_lightning as pl
from omegaconf import DictConfig

import experimenting


def load_model(load_path: str, module: str, **kwargs):
    """
    Main function to load a checkpoint.
    Args:
        load_path: path to the checkpoint directory
        module: python module (e.g., experimenting.agents.Base)
        kwargs: arguments to override while loading checkpoint

    Returns
        Lightning module loaded from checkpoint, if exists
    """
    print("Loading training")
    load_path = get_checkpoint_path(load_path)
    print("Loading from ... ", load_path)

    if os.path.exists(load_path):
        model = getattr(experimenting.agents, module).load_from_checkpoint(load_path, **kwargs)
    else:
        raise FileNotFoundError()

    return model


def get_checkpoint_path(checkpoint_dir: str) -> str:
    # CHECKPOINT file are
    if os.path.isdir(checkpoint_dir):
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.ckpt")))
        load_path = os.path.join(checkpoint_dir, checkpoints[0])
    else:
        raise Exception("Not checkpoint dir")
    return load_path


def instantiate_new_model(
    cfg: DictConfig, core: experimenting.dataset.BaseCore
) -> pl.LightningModule:
    """
    Instantiate new module from scratch using provided `hydra` configuration
    """
    model = getattr(experimenting.agents, cfg.training.module)(
        loss=cfg.loss,
        optimizer=cfg.optimizer,
        lr_scheduler=cfg.lr_scheduler,
        model_zoo=cfg.model_zoo,
        core=core,
        **cfg.training
    )
    return model
