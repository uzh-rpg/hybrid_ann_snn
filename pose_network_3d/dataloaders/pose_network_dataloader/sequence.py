import h5py
import torch
import numpy as np
import torchvision
import pdb
import math

from pathlib import Path
from os.path import join
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from dataloaders.data_representations import StackedHistogram, VoxelGrid
# from torch_impl.torch_utils.visualization import *

# todo check S16_session2_mov1 timestamps aren't properly generated
torch.manual_seed(0)

# todo crop data to 256x256 if run with the CNN results

class Movie(Dataset):

    # This class assumes the following structure in a sequence directory:
    #
    # event sequence name (e.g. S9_session1_mov1_2D_3D)
    # ├── estimates_2d (estimate of hybrid CNN-SNN network)
    # │   ├── camview2
    # │   └── camview3
    # ├── gt_2d (2D ground truths)
    # │   ├── camview2
    # │   └── camview3
    # └── gt_3d (3D ground truth)

    # The dataset is in the original resolution of 346x260. 

    # The hybrid models used to extract the esimates are: 
    # pretrained_weights_camview2 = /media/mathias/moar/asude/checkpoints/HYBRID_CNN_SNN_CONST_COUNT_10x10=100ms_tau3.0_output_decay0.8_camview2_6.pt
    # pretrained_weights_camview3 = /media/mathias/moar/asude/checkpoints/HYBRID_CNN_SNN_CONST_COUNT_10x10=100ms_tau3.0_output_decay0.8_camview3_2.pt
 
    # Both predictions and GT 2D are saved in the original resolution of the camera 346x260
    def __init__(self, movie_path, receptive_field):
        
        self.file_len = 0

        self.movie_path = movie_path
        self.receptive_field = receptive_field

        self.image_h, self.image_w, self.num_joints = 260, 346, 13
        self.image_w_new_0 = 34
        self.image_w_new_e = 290
        self.image_h_new_0 = 2
        self.image_h_new_e = 258

        self.shift = 1

        # h5 loading
        self.h5_dict = {}

        self.h5_opened = False

        with h5py.File(self.movie_path, 'r') as file:
            self.file_len = len(file['gt_3d'])


    def __open_h5(self):

        self.h5_dict = {
            'movie': h5py.File(self.movie_path, 'r'),
        }

        self.h5_opened = True

    def __close_h5(self):

        for i, h5 in self.h5_dict.items():
                h5.close()

        self.h5_opened = False

    def __len__(self):
        return int((self.file_len - self.receptive_field)/self.shift + 1)

    def __getitem__(self, idx: int):
        if not self.h5_opened:
            self.__open_h5()

        idx_start = idx
        idx_end = idx_start + self.receptive_field

        estimate_2d_cam3 = torch.from_numpy(self.h5_dict['movie']['estimates_2d/camview3'][idx_start:idx_end])
        estimate_2d_cam2 = torch.from_numpy(self.h5_dict['movie']['estimates_2d/camview2'][idx_start:idx_end])
                
        # flip image width for geometrical triangulation
        estimate_2d_cam3[:,:,1] = self.image_h - estimate_2d_cam3[:,:,1] 
        estimate_2d_cam2[:,:,1] = self.image_h - estimate_2d_cam2[:,:,1] 
        
        # Normalize joint cootdinates
        estimate_2d_cam3 = self.normalize_screen_coordinates(estimate_2d_cam3, self.image_w, self.image_h)
        estimate_2d_cam2 = self.normalize_screen_coordinates(estimate_2d_cam2, self.image_w, self.image_h)

        estimate_2d_cat = torch.cat((estimate_2d_cam3, estimate_2d_cam2), axis=-1)

        gt_2d_cam3 = torch.from_numpy(self.h5_dict['movie']['gt_2d/camview3'][idx_end-1])
        gt_2d_cam2 = torch.from_numpy(self.h5_dict['movie']['gt_2d/camview2'][idx_end-1])

        # flip image width for geometrical triangulation
        gt_2d_cam3[:,1] = self.image_h - gt_2d_cam3[:,1]
        gt_2d_cam2[:,1] = self.image_h - gt_2d_cam2[:,1]
        
        gt_2d_cat = torch.cat((gt_2d_cam3, gt_2d_cam2), axis=-1)

        gt_3d = torch.from_numpy(self.h5_dict['movie']['gt_3d'][idx_end-1])
        # gt_3d[:, 1:] -= gt_3d[:, :1] # remove global offset doesn't wotk well

        nan_mask = gt_3d.isfinite()

        sample = {'estimates_2d': estimate_2d_cat, 'gt_2d': gt_2d_cat, 'gt_3d': gt_3d, 'gt_3d_nan_mask':nan_mask}

        self.__close_h5()

        return sample

    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2
        
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        subtr = torch.broadcast_to(torch.tensor([1,h/w]), X.shape)
        return X/w*2 - subtr
