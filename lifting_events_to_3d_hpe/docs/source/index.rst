.. master_thesis documentation master file, created by
   sphinx-quickstart on Thu Aug 13 13:49:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********************************
Getting Started with Lifting Monocular Events to 3D Human Poses
**********************************
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   experimenting.dataset
   experimenting.agents
   experimenting.models


In brief
==========

* Train classification and reconstruction models based on ResNet34 and Resnet50
* Train 2D and 3D HPE models 
* Two datasets (DHP19, Event-H3m)
* Different event representations (constant-count, spatiotemporal voxelgrid)

Environment
============
Create a `virtualenv` environment from `requirements.txt`. 
Using pipenv:

```
pipenv install -r requirements.txt
pipenv shell
python -m pip install .
```

Data
============

.. mdinclude:: ../../scripts/dhp19/README.md

.. mdinclude:: ../../scripts/h3m/README.md

Model zoo
-----------
A model zoo of backbones and models for `constant_count` and `voxelgrid` trained
both with `DHP19` and `Events-H3m` is publicly accessible at [work in progress].

Agents
========
Train and evaluate for different tasks
If you want to launch an experiment with default parameters (backbone `ResNet50`, `DHP19` with `constant-count` representation, see the paper for details), you simply do (after setup and data):

Train
-------
A complete configuration is provided at `./confs/train/config.yaml`. In
particular, refer to `./confs/train/dataset/...` for dataset configuration
(including `path` specification), and to `./confs/train/training` for different
tasks. 

```
python train.py 
```


If you want to continue an ended experiment, you can set
`training.load_training` to `true` and provide a checkpoint path:

```
python train.py training.load_training=true training.load_path={YOUR_MODEL_CHECKPONT}
```

To initialize a model with a checkpoint of an ended experiments (load only the
model, not the trainer neither the optimizer status)

``` python train.py training.load_training=false training.load_path={YOUR_MODEL_CHECKPONT} ```

To train a margipose\_estimator agent:
```
python scripts/train.py training=margipose dataset=$DATASET training.model=$MODEL \
training.batch_size=$BATCH_SIZE training.stages=$N_STAGES
```
Supported dataset are: `constantcount_h3m`, `voxelgrid_h3m`, `constantcount_dhp19`, `voxelgrid_dhp19`


Test
-------

To evaluate a model, you can do the following (it generate a `results.json` file with the outputs):
```
python scripts/evaluate.py training.load_path={YOUR_MODEL_CHECKPOINT} dataset=$DATASET
```

To evaluate a model on per-movement protocol *for DHP19*, you can do:
```
python scripts/eveluate_dhp19_per_movement.py \
training.load_path={YOUR_MODEL_CHECKPOINT} dataset=$DATASET
```




