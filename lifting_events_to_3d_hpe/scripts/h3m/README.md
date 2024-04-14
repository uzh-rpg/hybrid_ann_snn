# Generate events-Human3.6m

## First steps

0. Request a LICENSE from academic porpoise for Human3.6m 
1. Download and extract human3.6m dataset. You can use the tool inside `human3.6m_downloader` or download `Videos` and `Raw_Angles` from https://github.com/facebookresearch/VideoPose3Dhttp://vision.imar.ro/human3.6m/description.php
NOTE: either case, you need to request a LICENSE from the authors
NOTE: you DON'T need `D3_Positions` or any other data from human3.6m. Joint positions are generated from raw values in the following steps. Remove any `D3_Positions` subdirectory from `MyPoseFeatures`

## Joints 
In order to generate good-quality labels, you need to use full joints positions (`D3_Positions`). These data are not distributed in human3.6m, which instead gives access to `RAW_Angles` only. We provide a matlab script to convert `Raw_angles` into `Full_D3_Positions` 

0. Get a valid `MATLAB` installation. We tested the script on MATLAB2021a
1. Launch Matlab into `Generate_Full_D3_Positions`. You should see `add paths for experiments`
2. Specify your folder director
3. Run script `generate_data_h36m.m`. After the process, you should have a `FULL_3D_Positions` in  your dataset

## Process

1. Use `event_library` generator script to generate `raw` events from `mp4` files:
```
python event_library/tools/generate.py frames_dir=path/to/dataset out_dir=out \
upsample=true emulate=true search=false representation=raw
```
2. Launch `prepare_data_h3m.py` to generate a `.npz` file containing FULL_D3_Positions
3. Launch `genearate_datasets.py` to generate `constant_count` frames and joints


## Docker
We provide a docker image at `https://hub.docker.com/repository/docker/gianscarpe/event-based-hpe` containing `human3.6m_downloader`, `event_library` and their dependencies. You still need to generate `FULL_D3_Positions` using a local MATLAB installation


## Copyrights
- `human3.6m_downloader` - https://github.com/kotaro-inoue/human3.6m_downloader
- `prepare_data_h3m.py` - Facebook research
- `Human3.6m` - https://github.com/facebookresearch/VideoPose3D
- `Event-library` - Gianluca Scarpellini (2020-2021)



