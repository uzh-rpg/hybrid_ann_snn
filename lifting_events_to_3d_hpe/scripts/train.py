import logging

import comet_ml
import hydra
from omegaconf import DictConfig

from experimenting.utils.trainer import HydraTrainer

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train/config.yaml')
def main(cfg: DictConfig) -> None:
    trainer = HydraTrainer(cfg)
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()
