import os
import re

import albumentations
import hydra
from omegaconf import DictConfig, ListConfig


def get_augmentation(augmentation_specifics: dict) -> albumentations.Compose:
    augmentations = []

    for _, aug_spec in augmentation_specifics['apply'].items():
        aug = hydra.utils.instantiate(aug_spec)

        augmentations.append(aug)

    return albumentations.Compose(augmentations)
