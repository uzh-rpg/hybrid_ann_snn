import json
import logging
import os

import hydra
from omegaconf import DictConfig

from experimenting.utils.evaluation_helpers import PerMovementEvaluator

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path='../confs/train/eval.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    result_path = os.path.join(cfg.load_path, cfg.result_file)
    evaluator = PerMovementEvaluator(cfg)
    results = evaluator.evaluate_per_movement()
    with open(result_path, 'w') as json_file:
        json.dump(results, json_file)


if __name__ == '__main__':
    main()
