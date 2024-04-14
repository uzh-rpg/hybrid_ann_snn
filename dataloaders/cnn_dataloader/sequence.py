import time
import weakref
import h5py
import torch
import numpy as np
import pdb
import math 

from pathlib import Path
from torch.nn.functional import pad
from torchvision.transforms.functional import crop
from os.path import join
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from dataloaders.data_representations import StackedHistogram, VoxelGrid

# to do check S16_session2_mov1 timestamps aren't properly generated
# torch.manual_seed(0)

class Movie(Dataset):

    # This class assumes the following structure in a sequence directory:
    #
    # event bin name (e.g. S9_session1_mov1_events_bin1)
    # ├── cam{}_events (where {} is camera id: 0-3)
    # │   ├── p (polarity)
    # │   ├── t (timestamp starting from 0)
    # │   ├── x
    # │   └── y
    # └── cam{}_ms_to_idx (where {} is camera id: 0-3)

    # label name (e.g. S9_session1_mov1_events_bin1_label)
    # └── XYZ
    #     ├── 3d_poses
    #     └── timestamps

    def __init__(self, movie_path, label_path, P_mat_path, cam_id, ev_repr, step_size, seq_len, num_bins, constant_count, constant_duration):

        self.movie_path = movie_path
        self.label_path = label_path

        self.image_h, self.image_w, self.num_joints, self.cam_id = 260, 346, 13, cam_id
        self.image_w_new_0 = 34
        self.image_w_new_e = 290
        self.image_h_new_0 = 2
        self.image_h_new_e = 258
        self.num_channels = 2

        self.constant_count = constant_count
        self.constant_duration = constant_duration

        self.ev_repr = ev_repr
        self.step_size = step_size
        self.output_seq_len = seq_len
        self.num_bins = num_bins
        self.constant_count_length = 7500
        self.label_step_size = int(self.step_size/10) # 10 labels in 100 ms bin

        # h5 loading
        self.h5_dict = {}

        if self.cam_id == 1:
            self.P_mat_cam = np.load(join(P_mat_path, 'P1.npy'))
        elif self.cam_id == 3:
            self.P_mat_cam = np.load(join(P_mat_path, 'P2.npy'))
        elif self.cam_id == 2:
            self.P_mat_cam = np.load(join(P_mat_path, 'P3.npy'))
        elif self.cam_id == 0:
            self.P_mat_cam = np.load(join(P_mat_path, 'P4.npy'))

        self.h5_opened = False

        self.h5_branch = 'cam{}_events'.format(self.cam_id)
        self.h5_branch_ms_to_idx = 'cam{}_ms_to_idx'.format(self.cam_id)

        with h5py.File(self.movie_path, 'r') as file:
            mov_ms_len = len(file[self.h5_branch_ms_to_idx])
            constant_count_start_us = file[self.h5_branch + '/t'][self.constant_count_length]
            constant_count_start_ms = constant_count_start_us/1e3
            self.constant_count_start_ms = math.ceil(constant_count_start_ms/100)*100
        
        with h5py.File(self.label_path, 'r') as file:
            lab_ms_len = int(file['XYZ']['timestamps'][-1]/1e3 + 10)

        self.seq_ms_len = min(mov_ms_len, lab_ms_len)

        if self.ev_repr == 'stacked_hist':
            self.event_repr = StackedHistogram(self.num_bins, self.image_h, self.image_w)
        elif self.ev_repr == 'voxel_grid':
            self.event_repr = VoxelGrid(self.num_bins, self.image_h, self.image_w, normalize = True)
        else:
            NotImplementedError

    def __open_h5(self):

        self.h5_dict = {
            'movie': h5py.File(self.movie_path, 'r'),
            'label': h5py.File(self.label_path, 'r')
        }

        self.h5_opened = True

    def __close_h5(self):

        for i, h5 in self.h5_dict.items():
                h5.close()

        self.h5_opened = False

    def __len__(self):
        if self.constant_count:
            return int(np.floor((self.seq_ms_len-self.constant_count_start_ms)/self.step_size))
        elif self.constant_duration:
            return int(np.floor((self.seq_ms_len-self.output_seq_len)/self.step_size)+1)

    def __getitem__(self, idx: int):
        if not self.h5_opened:
            self.__open_h5()
        
        if self.constant_count:

            idx_init = int(self.constant_count_start_ms/100)
            idx = idx + idx_init
            ms_start = int(idx*100 + (self.step_size-100)*(idx-idx_init))

            ms_end_cnn = int(ms_start)
            idx_end_cnn = int(self.h5_dict['movie'][self.h5_branch_ms_to_idx][ms_end_cnn])

            assert idx_end_cnn >= 7500
            events = {'x': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/x'][idx_end_cnn-self.constant_count_length:idx_end_cnn] - 1, dtype=torch.long),
                'y': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/y'][idx_end_cnn-self.constant_count_length:idx_end_cnn] - 1, dtype=torch.long),
                'p': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/p'][idx_end_cnn-self.constant_count_length:idx_end_cnn], dtype=torch.long),
                't': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/t'][idx_end_cnn-self.constant_count_length:idx_end_cnn], dtype=torch.long),}
                
            assert events['t'].shape[0] == self.constant_count_length
            assert events['x'].shape[0] == self.constant_count_length
            assert events['y'].shape[0] == self.constant_count_length
            assert events['p'].shape[0] == self.constant_count_length

            label_id = int(ms_end_cnn/10)-1
        
        elif self.constant_duration:
            ms_start = int(idx*self.step_size)
            idx_start = int(self.h5_dict['movie'][self.h5_branch_ms_to_idx][ms_start])

            ms_end = int(ms_start + self.output_seq_len)
            if ms_end == self.seq_ms_len:
                idx_end = -1
            else:
                idx_end = int(self.h5_dict['movie'][self.h5_branch_ms_to_idx][ms_end])
                assert idx_start <= idx_end
            assert ms_start < ms_end
            assert ms_end <= len(self.h5_dict['movie'][self.h5_branch_ms_to_idx])

            if idx_start == idx_end:
                ev_repr_final = torch.zeros(self.num_channels*self.num_bins, self.image_h_new_e-self.image_h_new_0, self.image_w_new_e-self.image_w_new_0)
            else:
                # subtract 1 from x and y values for range [0,255]
                events = {'x': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/x'][idx_start:idx_end] - 1, dtype=torch.long),
                        'y': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/y'][idx_start:idx_end] - 1, dtype=torch.long),
                        'p': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/p'][idx_start:idx_end], dtype=torch.long),
                        't': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/t'][idx_start:idx_end], dtype=torch.long),}

            # label at 100 Hz (10 ms) (label rate recording)
            label_id = (idx+1)*self.label_step_size - 1
        
        ev_repr = self.event_repr.construct(events['x'], events['y'], events['p'], events['t'])
        ev_repr_cat = torch.cat((ev_repr[:,0], ev_repr[:,1]), dim=0) 
        ev_repr_final = ev_repr_cat[:, self.image_h_new_0:self.image_h_new_e, self.image_w_new_0:self.image_w_new_e]     

        assert ev_repr_final.shape[0] == self.num_channels*self.num_bins 

        xyz_landmarks = self.h5_dict['label']['XYZ']['3d_poses'][label_id]
        label_timestamp_ms = self.h5_dict['label']['XYZ']['timestamps'][label_id]/1e3 # ms 

        # convert 3D landmarks to 2D image
        uv_landmarks, mask = self.xyz_to_uv(xyz_landmarks)
        label_image = self.uv_to_image(uv_landmarks, mask)
        label_final = torch.from_numpy(label_image)

        uv_landmarks_float = self.set_back_to_nan(uv_landmarks, mask)
        landmarks_final = torch.from_numpy(uv_landmarks_float)

        # todo make sure to apply a gaussian kernel over the constant count labels
        sample = {'event_representation': ev_repr_final, 'label': label_final, 'landmarks': landmarks_final,
                'filename': Path(self.label_path).stem, 'timestamp': label_timestamp_ms}

        self.__close_h5()

        return sample

    def xyz_to_uv(self, sample):

        vicon_xyz_homog = np.concatenate([sample, np.ones([1, self.num_joints])], axis=0)
        coord_pix_homog = np.matmul(self.P_mat_cam, vicon_xyz_homog)
        coord_pix_homog_norm = coord_pix_homog / coord_pix_homog[-1]
        u = coord_pix_homog_norm[0]
        v = self.image_h - coord_pix_homog_norm[1]

        # mask is used to make sure that pixel positions are in frame range.
        mask = np.ones(u.shape).astype(np.float32)
        mask[np.isnan(u)] = 0
        mask[np.isnan(v)] = 0
        mask[u >= self.image_w_new_e] = 0
        mask[u < self.image_w_new_0] = 0
        mask[v >= self.image_h_new_e] = 0
        mask[v < self.image_h_new_0] = 0

        u = u-self.image_w_new_0
        v = v-self.image_h_new_0

        # pixel coordinates
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        return (u, v), mask

    def uv_to_image(self, pix_locs, mask):
        label_image = np.zeros((self.num_joints, self.image_h_new_e-self.image_h_new_0, self.image_w_new_e-self.image_w_new_0))

        for fmidx, zipd in enumerate(zip(pix_locs[1], pix_locs[0], mask)):
            if zipd[2] == 1:  # write joint position only when projection within frame boundaries
                label_image[fmidx, zipd[0], zipd[1]] = 1

        return label_image

    def set_back_to_nan(self, uv_landmarks, mask):
        # set back to NaN where mask is zero for mpjpe metric
        y_2d = np.asarray(uv_landmarks)
        y_2d_float = y_2d.astype(np.float)

        y_2d_float[0][mask == 0] = np.nan
        y_2d_float[1][mask == 0] = np.nan

        return y_2d_float

