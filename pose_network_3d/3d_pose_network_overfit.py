import pdb
import torch
import numpy as np
import torch.nn as nn
import h5py

from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from common.ranger import Ranger

from configs.args import *
from torch_utils.utils import *
from torch_utils.network_initialization import *
from torch_utils.visualization import *
from models.pose_network_3D import TemporalModelOptimized1f
from spikingjelly.clock_driven import functional

def normalize_screen_coordinates(X, w, h, device): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    subtr = torch.broadcast_to(torch.tensor([1,h/w], device=device), X.shape)
    return X/w*2 - subtr

def calc_3D_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


if __name__ == '__main__':
    save_runs = True

    _seed_ = 2022
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True    
    torch.backends.cudnn.allow_tf32 = True

    # SET DEVICE
    device = 'cuda:1'

    h5 = h5py.File('/media/mathias/moar/asude/3d_pose_network/S10_session2_mov1_2D_3D.h5','r')

    inputs = torch.from_numpy(h5['2d_pose_estimates'][:486]).to(device).unsqueeze(dim=0)
    inputs = torch.cat((inputs[:,:243], inputs[:,243:]), dim=0)

    # inputs = normalize_screen_coordinates(inputs, 256, 256, device)

    label1 = torch.from_numpy(h5['3d_pose_gt'][242]).to(device).unsqueeze(dim=0)
    label2 = torch.from_numpy(h5['3d_pose_gt'][485]).to(device).unsqueeze(dim=0)

    gt_3d = torch.cat((label1, label2), dim=0).unsqueeze(dim=1)

    # todo write original x,y values
    
    #  remove global offset?
    # gt_3d[:, :, 1:] -= gt_3d[:, :, :1]  # Remove global offset, but keep trajectory in first position

    #  normalize 2D coordinates to [-1,1]

    num_joints_in = 13
    in_features = 2 # for
    num_joints_out = 13
    filter_widths = '3,3,3,3,3'
    filter_widths = [int(x) for x in filter_widths.split(',')]

    model = TemporalModelOptimized1f(num_joints_in=num_joints_in, in_features=in_features, num_joints_out=num_joints_out, filter_widths=filter_widths).to(device)
    
    lr_value = 0.0001    

    # optimizer = Ranger(model_pos_train.parameters(), lr=lr)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=args.epochs) # LR COSINE 
    
    optimizer = Adam(model.parameters(), lr=lr_value)

    if save_runs: 
        comment = '_Overfit_3D_pose_network_camview3_Adam_normalize_input_' 
        tb = SummaryWriter(comment=comment)

    for count in range(100000):
        print(count)

        optimizer.zero_grad()

        pred_3d = model(inputs.float())
        loss = calc_3D_mpjpe(pred_3d, gt_3d)

        loss.backward()
        optimizer.step()

        # if count%100:
        tb.add_scalar('Loss (3D MPJPE)', loss.item(), count)
    
    # todo check mpjpe function runs correctly 
