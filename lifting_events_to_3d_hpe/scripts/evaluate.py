import json
import logging
import os
import pdb

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from experimenting import agents
from experimenting.utils.trainer import HydraTrainer

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train/eval.yaml')
def main(cfg: DictConfig) -> None:
    trainer = HydraTrainer(cfg)
    trainer.test()


if __name__ == '__main__':
    main()
