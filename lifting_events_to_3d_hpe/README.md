[![Build Status](https://travis-ci.com/gianscarpe/event-based-monocular-hpe.svg?branch=master)](https://travis-ci.com/gianscarpe/event-based-monocular-hpe)
[![Documentation
Status](https://readthedocs.org/projects/event-camera/badge/?version=latest)](https://event-camera.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/gianscarpe/event-based-monocular-hpe/badge.svg?branch=master)](https://coveralls.io/github/gianscarpe/event-based-monocular-hpe?branch=master)
# Lifting Monocular Events to 3D Human Poses

* Train classification models based on ResNet18, Resnet34, ...
* Train 3D reconstruction models
* Dataset adpatation for DHP19 dataset
* Generate events from events dataset with different frames representations
  (constant-count, spatiotemporal voxelgrid)

## Table of contents
- [Environment](#environment)
- [Data](#data)
- [Model zoo](#model-zoo)
- [Agents](#agents)


## Environment
Create a `virtualenv` environment from `requirements.txt`. 
Using pipenv:

```
pipenv install -r requirements.txt
pipenv shell
python -m pip install .
```

## Data


### DHP19
Follow DHP19 guide at `scripts/dhp19/README.md`

### Events-H3m
Follow the guide at `scripts/h3m/README.md`

### Model zoo
A model zoo of backbones and models for `constant_count` and `voxelgrid` trained
both with `DHP19` and `Events-H3m` is publicly accessible at [this link](https://drive.google.com/drive/folders/1b5BTt4A8kRGPAVwZ2HY1RKoMQ54lPsGZ?usp=sharing)

## Agents

### Train and evaluate for different tasks
If you want to launch an experiment with default parameters (backbone `ResNet50`, `DHP19` with `constant-count` representation, see the paper for details), you simply do (after setup and data):

```
python train.py 
```

A complete configuration is provided at `./confs/train/config.yaml`. In
particular, refer to `./confs/train/dataset/...` for dataset configuration
(including `path` specification), and to `./confs/train/training` for different
tasks. 

If you want to continue an ended experiment, you can set
`training.load_training` to `true` and provide a checkpoint path:

```
python train.py training.load_training=true training.load_path={YOUR_MODEL_CHECKPONT}
```

To continue a previous experiment:
```
python train.py training.load_training=true training.load_path={YOUR_MODEL_CHECKPONT}
```

To train a margipose\_estimator agent:
```
python scripts/train.py training=margipose dataset=$DATASET training.model=$MODEL training.batch_size=$BATCH_SIZE training.stages=$N_STAGES
```
Supported dataset are: `constantcount_h3m`, `voxelgrid_h3m`, `constantcount_dhp19`, `voxelgrid_dhp19`
To evaluate a model, you can use:
```
python scripts/eveluate.py training.load_path={YOUR_MODEL_CHECKPOINT}
```

### Test
You can test your models using our multi-movement evaluation script. The tool
generates a `result.json` file in the provided checkpoint path.
```
python evaluate_dhp19.py training={TASK} dataset={DATASET_REPRESENTATION} load_path={YOUR_MODEL_CHECKPOINT}
```

This framework is intended to be fully extensible. It's based upon
`pytorch_lighting` [[1]](#1) and `hydra` configuration files.

## References
<a id="1">[1]</a> 
Falcon, WA and .al (2019). 
PyTorch Lightning
GitHub. Note: https://github.com/PyTorchLightning/pytorch-lightning

