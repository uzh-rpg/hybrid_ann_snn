# COMMONLY USED COMMANDS

This file is a cheat sheet to help with running files.

## H36M 

### CNN Training 

`python3 train/train_cnn_model_h36m.py --config_path=/home/asude/master_thesis/configs/cnn_config_h36m.ini --no-save_params --output_path='' --device='cuda:0' --tb_name='test'`

## DHP19

### 3D Pose Network 
`python3 geometric_triangulation.py --config_path=/home/asude/thesis_final/master_thesis/pose_network_3d/pose_network_3d.ini --device=cuda:1 --pretrained_dict_path='' --txt_path='double_check.txt'`