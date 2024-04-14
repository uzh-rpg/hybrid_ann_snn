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

    # This class assumes the follow  ing structure in a sequence directory:
    #
    # event sequence name (e.g. S9_session1_mov1_events)
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

    def __init__(self, movie_path, label_path, P_mat_path, ev_repr, bin_length_per_stack, step_size, cam_id, seq_len, cnn_input, constant_count, constant_duration):

        self.movie_path = movie_path
        self.label_path = label_path
        self.cnn_input = cnn_input
        self.constant_count = constant_count
        self.constant_duration = constant_duration
        self.constant_count_length = 7500

        if self.cnn_input:
            assert self.constant_count != self.constant_duration

        self.image_h, self.image_w, self.num_joints, self.cam_id = 260, 346, 13, cam_id
        self.image_w_new_0 = 34
        self.image_w_new_e = 290
        self.image_h_new_0 = 2
        self.image_h_new_e = 258

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

        # take 7500th timestamp to determine sample start ms
        with h5py.File(self.movie_path, 'r') as file:
            mov_ms_len = len(file[self.h5_branch_ms_to_idx])
            if self.constant_count:
                constant_count_start_us = file[self.h5_branch + '/t'][self.constant_count_length]
                constant_count_start_ms = constant_count_start_us/1e3
                self.constant_count_start_ms = math.ceil(constant_count_start_ms/100)*100

        # some labels are shoerter than sequence length which might cause problems
        with h5py.File(self.label_path, 'r') as file:
            lab_ms_len = int(file['XYZ']['timestamps'][-1]/1e3 + 10)

        self.seq_ms_len = min(mov_ms_len, lab_ms_len) 

        self.num_channels = 2
        self.output_seq_len = seq_len # sequence length in ms 
        self.step_size = step_size # how much to step to get the next sequence in ms
        self.label_step_size = int(self.step_size/10) # how much label should step for next sequence labels
        self.bin_length_per_stack = bin_length_per_stack # duration of events per stack of histogram 
        self.num_bins = int(self.output_seq_len/self.bin_length_per_stack) # count of bins per sequence

        self.label_rate = int(self.output_seq_len/10) # label rate in sequence, max allowed is every 10 ms 

        assert self.step_size == 100, "Step size other than 100 are not supported at the moment."

        if ev_repr == 'voxel_grid':
            self.event_repr = VoxelGrid(self.num_bins, self.image_h, self.image_w, normalize = True)
            self.event_repr_cnn = VoxelGrid(10, self.image_h, self.image_w, normalize = True)
        elif ev_repr == 'stacked_hist':
            self.event_repr = StackedHistogram(self.num_bins, self.image_h, self.image_w)
            self.event_repr_cnn = StackedHistogram(10, self.image_h, self.image_w)
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
            return int(np.floor((self.seq_ms_len-self.constant_count_start_ms-self.step_size)/self.step_size))
        else:
            return int(np.floor((self.seq_ms_len-self.output_seq_len)/self.step_size)+1)


    def __getitem__(self, idx: int):
        if not self.h5_opened:
            self.__open_h5()

        if self.constant_count: # find first index if self.constant_count is enabled 
            idx = idx + int(self.constant_count_start_ms/100)
        
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

        # CONSTRUCT SNN SEQUENCE INPUT
        if idx_start == idx_end:
            ev_repr_final = torch.zeros(self.num_bins, self.num_channels, self.image_h_new_e-self.image_h_new_0, self.image_w_new_e-self.image_w_new_0)
        else:
            # subtract 1 from x and y values for range [0,255]
            events = {'x': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/x'][idx_start:idx_end] - 1, dtype=torch.long),
                      'y': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/y'][idx_start:idx_end] - 1, dtype=torch.long),
                      'p': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/p'][idx_start:idx_end], dtype=torch.long),
                      't': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/t'][idx_start:idx_end], dtype=torch.long),}

            ev_repr = self.event_repr.construct(events['x'], events['y'], events['p'], events['t'])
            ev_repr_reshape = ev_repr[:, :, self.image_h_new_0:self.image_h_new_e, self.image_w_new_0:self.image_w_new_e]
            ev_repr_final = ev_repr_reshape

        assert ev_repr_final.shape[0] == self.num_bins

        # CONSTRUCT CNN INPUT 
        if self.cnn_input:
            if self.constant_duration: # todo start cnn input from [0,100] and snn sequence from [100,200]
                ms_start_cnn = int(ms_start - self.output_seq_len)
                ms_end_cnn = int(ms_start)

                if ms_start_cnn >= 0:
                    idx_start_cnn = int(self.h5_dict['movie'][self.h5_branch_ms_to_idx][ms_start_cnn])
                    idx_end_cnn = int(self.h5_dict['movie'][self.h5_branch_ms_to_idx][ms_end_cnn])

                    if idx_start_cnn == idx_end_cnn:
                        ev_repr_cnn_final = torch.zeros(10*self.num_channels, self.image_h_new_e-self.image_h_new_0, self.image_w_new_e-self.image_w_new_0) #TODO
                
                    else:
                        events_cnn = {'x': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/x'][idx_start_cnn:idx_end_cnn] - 1, dtype=torch.long),
                            'y': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/y'][idx_start_cnn:idx_end_cnn] - 1, dtype=torch.long),
                            'p': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/p'][idx_start_cnn:idx_end_cnn], dtype=torch.long),
                            't': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/t'][idx_start_cnn:idx_end_cnn], dtype=torch.long),}

                        ev_repr_cnn = self.event_repr.construct(events_cnn['x'], events_cnn['y'], events_cnn['p'], events_cnn['t'])
                        ev_repr_cnn_cat = torch.cat((ev_repr_cnn[:,0], ev_repr_cnn[:,1]), dim=0)
                        ev_repr_cnn_final = ev_repr_cnn_cat[:, self.image_h_new_0:self.image_h_new_e, self.image_w_new_0:self.image_w_new_e]
                else: 
                    ev_repr_cnn_final = torch.zeros(10*self.num_channels, self.image_h_new_e-self.image_h_new_0, self.image_w_new_e-self.image_w_new_0) #TODO
        
            elif self.constant_count:
                ms_end_cnn = int(ms_start)
                idx_end_cnn = int(self.h5_dict['movie'][self.h5_branch_ms_to_idx][ms_end_cnn])

                assert idx_end_cnn >= 7500
                events_cnn = {'x': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/x'][idx_end_cnn-self.constant_count_length:idx_end_cnn] - 1, dtype=torch.long),
                    'y': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/y'][idx_end_cnn-self.constant_count_length:idx_end_cnn] - 1, dtype=torch.long),
                    'p': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/p'][idx_end_cnn-self.constant_count_length:idx_end_cnn], dtype=torch.long),
                    't': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/t'][idx_end_cnn-self.constant_count_length:idx_end_cnn], dtype=torch.long),}
                
                assert events_cnn['t'].shape[0] == self.constant_count_length
                assert events_cnn['x'].shape[0] == self.constant_count_length
                assert events_cnn['y'].shape[0] == self.constant_count_length
                assert events_cnn['p'].shape[0] == self.constant_count_length

                ev_repr_cnn = self.event_repr_cnn.construct(events_cnn['x'], events_cnn['y'], events_cnn['p'], events_cnn['t'])
                ev_repr_cnn_cat = torch.cat((ev_repr_cnn[:,0], ev_repr_cnn[:,1]), dim=0) 
                ev_repr_cnn_final = ev_repr_cnn_cat[:, self.image_h_new_0:self.image_h_new_e, self.image_w_new_0:self.image_w_new_e]     
            
                if self.output_seq_len == 100:
                    ms_end_cnn2 = int(ms_start+100)
                    if ms_end_cnn2 == len(self.h5_dict['movie'][self.h5_branch_ms_to_idx]):
                        ms_end_cnn2 -= 1
                    idx_end_cnn2 = int(self.h5_dict['movie'][self.h5_branch_ms_to_idx][ms_end_cnn2])

                    assert idx_end_cnn >= 7500
                    events_cnn2 = {'x': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/x'][idx_end_cnn2-self.constant_count_length:idx_end_cnn2] - 1, dtype=torch.long),
                        'y': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/y'][idx_end_cnn2-self.constant_count_length:idx_end_cnn2] - 1, dtype=torch.long),
                        'p': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/p'][idx_end_cnn2-self.constant_count_length:idx_end_cnn2], dtype=torch.long),
                        't': torch.tensor(self.h5_dict['movie'][self.h5_branch + '/t'][idx_end_cnn2-self.constant_count_length:idx_end_cnn2], dtype=torch.long),}
                    
                    assert events_cnn2['t'].shape[0] == self.constant_count_length
                    assert events_cnn2['x'].shape[0] == self.constant_count_length
                    assert events_cnn2['y'].shape[0] == self.constant_count_length
                    assert events_cnn2['p'].shape[0] == self.constant_count_length

                    ev_repr_cnn2 = self.event_repr_cnn.construct(events_cnn2['x'], events_cnn2['y'], events_cnn2['p'], events_cnn2['t'])
                    ev_repr_cnn_cat2 = torch.cat((ev_repr_cnn2[:,0], ev_repr_cnn2[:,1]), dim=0) 
                    ev_repr_cnn_final2 = ev_repr_cnn_cat2[:, self.image_h_new_0:self.image_h_new_e, self.image_w_new_0:self.image_w_new_e]     
                    
                    ev_repr_cnn_final = torch.cat((ev_repr_cnn_final.unsqueeze(dim=0), ev_repr_cnn_final2.unsqueeze(dim=0)), dim=0)
            else: 
                assert NotImplementedError

        # CONSTRUCT SEQUENCE LABELS 
        label_start_id = idx*self.label_step_size
        label_end_id = label_start_id + self.label_rate
        label_3d = self.h5_dict['label']['XYZ']['3d_poses'][label_start_id:label_end_id]
        label_3d_final = torch.from_numpy(label_3d)

        assert label_3d_final.shape[0] == self.label_rate
        
        label_2d, mask = self.xyz_to_uv(label_3d)
        label_final = torch.from_numpy(self.uv_to_image(label_2d, mask))

        label_2d_float = self.set_back_to_nan(label_2d, mask)
        landmarks_final = torch.from_numpy(label_2d_float)

        # CONSTRUCT CNN LABELS
        if self.cnn_input:
            if idx == 0: # todo fix 
                label_init_timestamp = 0
                xyz_landmarks_init = self.h5_dict['label']['XYZ']['3d_poses'][0]
            else:
                label_init_timestamp = self.h5_dict['label']['XYZ']['timestamps'][label_start_id-1]
                xyz_landmarks_init = self.h5_dict['label']['XYZ']['3d_poses'][label_start_id-1]
                
            # convert 3D landmarks to 2D groundtruth heatmap
            uv_landmarks_init, mask_init = self.xyz_to_uv(xyz_landmarks_init)
            label_image_init = self.uv_to_image(uv_landmarks_init, mask_init)
            label_final_init = torch.from_numpy(label_image_init)
            uv_landmarks_float_init = self.set_back_to_nan(uv_landmarks_init, mask_init)
        
        if self.cnn_input:
            sample = {'event_representation': ev_repr_final, 'label': label_final, 'landmarks': landmarks_final, 
                        'cnn_input':ev_repr_cnn_final, 'cnn_label':label_final_init, 'cnn_landmarks': uv_landmarks_float_init,
                        'ms_start': ms_start, 'ms_end': ms_end, 'label_3d':label_3d_final}
        else:            
            sample = {'event_representation': ev_repr_final, 'label': label_final, 'landmarks': landmarks_final, 
                        'ms_start': ms_start, 'ms_end': ms_end, 'label_3d':label_3d_final}

        self.__close_h5()

        return sample

    def xyz_to_uv(self, sample):
        if len(sample.shape) == 3:
            vicon_xyz_homog = np.concatenate([sample, np.ones([self.label_rate, 1, self.num_joints])], axis=1)
            coord_pix_homog = np.matmul(self.P_mat_cam, vicon_xyz_homog)
            coord_div = coord_pix_homog[:, -1, :]
            # coord_div = torch_impl.squeeze(coord_pix_homog[:, -1], dim=1)
            for img_idx in range(self.label_rate):
                coord_pix_homog[img_idx] = coord_pix_homog[img_idx] / coord_div[img_idx]
            u = coord_pix_homog[:, 0]
            v = self.image_h - coord_pix_homog[:, 1]
        else:
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

        # pixel coordinates, note: int floors the values
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        return (u, v), mask

    def uv_to_image(self, pix_locs, mask):
        if len(np.shape(pix_locs)) == 3:
            label_image = np.zeros((self.label_rate, self.num_joints, self.image_h_new_e-self.image_h_new_0, self.image_w_new_e-self.image_w_new_0))

            for lb_idx in range(self.label_rate):
                for fmidx, zipd in enumerate(zip(pix_locs[1][lb_idx], pix_locs[0][lb_idx], mask[lb_idx])):
                    if zipd[2] == 1:  # write joint position only when projection within frame boundaries
                        label_image[lb_idx, fmidx, zipd[0], zipd[1]] = 1
        else:
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

if __name__ == '__main__':
    # from torch_utils.visualization import plot_batch_label_pred_tb
    from provider import get_dataset
    # from .path_locs import *
    import matplotlib.pyplot as plt
    torch.manual_seed(0)
    np.random.seed(0)
    import configparser

    config_path = '/home/asude/thesis_final/master_thesis/configs/hybrid_cnn_snn_config.ini'
    configfile = configparser.ConfigParser()
    configfile.read(config_path)
    # get config
    dataloader_config = configfile['dataloader_parameters']

    dataset_provider = get_dataset(dataloader_config)
    dhp19_dataset_train = dataset_provider.get_train_dataset()
    dhp19_dataset_test = dataset_provider.get_test_dataset()
    
    device='cpu'
    params = {'batch_size': 2,
              'shuffle': False,
              'num_workers': 0}

    train_loader = DataLoader(dhp19_dataset_train, **params)
    test_loader = DataLoader(dhp19_dataset_test, **params)

    # for i_batch, sample_batched in enumerate(train_loader):
    #     print(i_batch)
    #     inputs = sample_batched['event_representation'].to(device)
    #     cnn_input = sample_batched['cnn_input'].to(device)

    for i_batch, sample_batched in enumerate(test_loader):
        print(i_batch)
        inputs = sample_batched['event_representation'].to(device)
        cnn_input = sample_batched['cnn_input'].to(device)

        # landmarks = sample_batched['landmarks'].to(device)
        # labels = sample_batched['label'].to(device)
#         pred_landmarks = torch.randint(low=0, high=256, size=(8, 2, 10, 13))


# #         plot_batch_label_pred_tb(sample_batched, sample_batched['label_init'], points='all', grid_shape=(2,4))
#         plot_batch_label_pred_tb(sample_batched, pred_landmarks, points='all', grid_shape=(2,4))

#         plt.show()

#         pdb.set_trace()

#         if i_batch == 20:
#             break

#         print('Batch number :', i_batch)

    # for i_batch, sample_batched in enumerate(test_loader):
    #     # inputs = sample_batched['event_representation'].to(device)
    #     labels = sample_batched['label_init'].to(device)
    #     # landmarks = sample_batched['landmarks'].to(device)
    #     print('Batch number :', i_batch)

    # for i_batch, sample_batched in enumerate(train_loader):
    #     # inputs = sample_batched['event_representation'].to(device)
    #     labels = sample_batched['label_init'].to(device)
    #     # landmarks = sample_batched['landmarks'].to(device)
    #     print('Batch number :', i_batch)


#     ##### Analyze batch loading time #####
#
#     t0 = time.time() # analyze time for processing
#     for i_batch, sample_batched in enumerate(train_loader):
#         print('Batch {}'.format(i_batch))
# #         print('Open files {}'.format(len(psutil.Process().open_files()))) # prints number of files open
#         if i_batch == 79:
#             break
#     print(time.time()-t0)

#################################################################
#
# #     ### VISUALIZE DATASET ###
# #
#     from torch_impl.utils.data import DataLoader
#     from torch_utils.plotting import plot_labeled_batch
#     from dataset.provider import get_dataset
#     from dataset.visualization import plot_batch_from_dataloader, plot_samples_from_dataset
#
#     dhp19_dataset = get_dataset(path_dir, P_mat_dir)
#     train_set = dhp19_dataset.get_train_dataset()
#
#     dataloader = DataLoader(
#         train_set,
#         shuffle=True,
#         num_workers=0,
#         batch_size=8
#     )
#
#     plot_batch_from_dataloader(dataloader, 0)
#
#     plot_samples_from_dataset(3, 2, train_set)
