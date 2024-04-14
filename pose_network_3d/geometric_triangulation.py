import pdb
import torch
import numpy as np
import torch.nn as nn
import h5py

from tqdm import tqdm
from datetime import datetime
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from os.path import join

from configs.args import *
from torch_utils.utils import *
from torch_utils.ranger import Ranger
from torch_utils.utils_3d import *
from torch_utils.network_initialization import *
from torch_utils.visualization import *
from pose_network_3d.models.pose_network_3D import TemporalModelOptimized1f
from spikingjelly.clock_driven import functional
from pose_network_3d.dataloaders.pose_network_dataloader.provider import get_dataset

if __name__ == '__main__':
    flags, configfile = FLAGS(inference=True)
    print('Warning this file is model independent and only supports receptive_field = 1')

    # get config
    dataloader_config = configfile['dataloader_parameters']
    training_config = configfile['training_parameters']
    inference_config = configfile['inference_parameters']

    _seed_ = 2022
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True    
    torch.backends.cudnn.allow_tf32 = True

    # SET DEVICE
    device = torch.device(flags.device)

    P_mat_dir = '/home/rslsync/Resilio Sync/DHP19/P_matrices/'
    P_mat_cam2 = torch.from_numpy(np.load(join(P_mat_dir,'P2.npy'))).to(device)
    P_mat_cam3 = torch.from_numpy(np.load(join(P_mat_dir,'P3.npy'))).to(device)
    
    P_mats = [P_mat_cam2, P_mat_cam3]
    cameras_pos = np.load(join(P_mat_dir,'camera_positions.npy'))

    point0 = torch.from_numpy(np.stack((cameras_pos[1], cameras_pos[2]))).to(device)


    # PREPARE DATA LOADER
    dhp19_dataset = get_dataset(dataloader_config['dataset_path'], 1)
    train_set = dhp19_dataset.get_train_dataset() 
    val_set = dhp19_dataset.get_test_dataset() 
    print(val_set.__len__())

    params = {'batch_size': training_config.getint('batch_size'),
            'num_workers': training_config.getint('num_workers'),
            'pin_memory': True}

    train_loader = DataLoader(
        train_set,
        shuffle = True,
        **params
    )
    train_len = train_loader.__len__()

    val_loader = DataLoader(
        val_set,
        shuffle = False,
        **params, 
    )
    val_len = val_loader.__len__()

    mpjpe_cam3 = 0
    mpjpe_cam2 = 0
    mpjpe_3d_geo = 0
    mpjpe_3d_new = 0
    mpjpe_3d_intersection = 0
    counter = 0

    tot_batch_size = 0
    for batch_id, sample_batched in tqdm(enumerate(val_loader), total=val_len, desc='Total samples: '):

        inputs = sample_batched['estimates_2d'].to(device)
        gt_3d = sample_batched['gt_3d'].to(device)
        gt_2d = sample_batched['gt_2d'].to(device)
        gt_3d_mask = sample_batched['gt_3d_nan_mask'].to(device)

        batch_size = inputs.shape[0]

        # calculate 2d mpjpe per batch - create empty array same length as batch size
        mpjpe_per_batch3 = torch.zeros(inputs.shape[0])
        mpjpe_per_batch2 = torch.zeros(inputs.shape[0])

        xyz_cam3 = torch.zeros(inputs.shape[0], inputs.shape[1], 13, 3, device=device)
        xyz_cam2 = torch.zeros(inputs.shape[0], inputs.shape[1], 13, 3, device=device)

        mpjpe_3d_intersection = torch.zeros(inputs.shape[0], inputs.shape[1], 13, 3, device=device)
        for b in range(inputs.shape[0]): # iterate over batch
            dist_2d = torch.linalg.norm((inputs[b,0,:,:2] - gt_2d[b,:,:2]), axis=-1)
            mpjpe_per_batch3[b] = torch.nanmean(dist_2d)

            dist_2d = torch.linalg.norm((inputs[b,0,:,2:] - gt_2d[b,:,2:]), axis=-1)
            mpjpe_per_batch2[b] = torch.nanmean(dist_2d)

            for f in range(inputs.shape[1]): # iterate over samples per batch (currently only supports receptive field == 1)
                input3 = torch.cat((inputs[b,f,:,0].unsqueeze(dim=1).T, inputs[b,f,:,1].unsqueeze(dim=1).T), dim=0)
                input2 = torch.cat((inputs[b,f,:,2].unsqueeze(dim=1).T, inputs[b,f,:,3].unsqueeze(dim=1).T), dim=0)
                xyz_cam3[b,f] = project_uv_xyz_cam(input3.double(), P_mat_cam2, device)
                xyz_cam2[b,f] = project_uv_xyz_cam(input2.double(), P_mat_cam3, device)

                xyz_cam3_temp = xyz_cam3[b,f].unsqueeze(dim=1)
                xyz_cam2_temp = xyz_cam2[b,f].unsqueeze(dim=1)

                pred_3d = torch.zeros((13,3), device=device)

                point1 = torch.cat((xyz_cam3_temp, xyz_cam2_temp), dim=1)
                for joint_idx in range(13):
                    intersection = find_intersection(point0, point1[joint_idx], device)
                    mpjpe_3d_intersection[b,f,joint_idx,:] = intersection[0]
                    pred_3d[joint_idx] = intersection[0]

                mpjpe_3d_joints = torch.linalg.norm((gt_3d[b] - pred_3d), axis=-1)
                mpjpe_3d_sample = torch.nanmean(mpjpe_3d_joints)
                mpjpe_3d_geo += mpjpe_3d_sample
                counter += 1

        mpjpe_3d_new += calc_3D_mpjpe(mpjpe_3d_intersection, gt_3d, gt_3d_mask)*batch_size

        mpjpe_cam3 += torch.nanmean(mpjpe_per_batch3)*batch_size # add average per batch
        mpjpe_cam2 += torch.nanmean(mpjpe_per_batch2)*batch_size

        tot_batch_size += batch_size

    with open(flags.txt_path, 'w') as f:
        f.write('%s:%s\n' % ('2D MPJPE camview 3', mpjpe_cam3.item()/tot_batch_size))
        f.write('%s:%s\n' % ('2D MPJPE camview 2', mpjpe_cam2.item()/tot_batch_size))
        f.write('%s:%s\n' % ('3D MPJPE geometrical', mpjpe_3d_geo.item()/counter))   
        f.write('%s:%s\n' % ('3D MPJPE geometrical with new function', mpjpe_3d_new.item()/tot_batch_size))


        
