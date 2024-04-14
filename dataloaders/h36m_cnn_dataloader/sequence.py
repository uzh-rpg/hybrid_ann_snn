import time
import weakref
import h5py
import torch
import numpy as np
import pdb
import math 
import os

from glob import glob
from pathlib import Path
from os.path import join
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.io import read_image
from dataloaders.data_representations import StackedHistogram, VoxelGrid

class Movie(Dataset):

    def __init__(self, dataset_path, subject_id, cam_id, seq_path, frame_interval, img_type, label_path=None):

        self.subject_name = subject_id
        self.cam_id = cam_id
        self.seq_path = seq_path
        self.frame_interval = frame_interval # ms
        self.constant_count_length = 7500
        
        self.img_type = img_type
        assert self.img_type == 'grayscale' or self.img_type == 'RGB' or self.img_type == 'constcount'
        self.image_w, self.image_h, self.num_joints = 320, 256, 13

        if self.img_type == 'constcount': # constant count events
            self.movie_path = join(seq_path, 'events.h5') 
            assert Path(self.movie_path).exists()
            with h5py.File(self.movie_path, 'r') as file:
                self.mov_ms_len = len(file['ms_to_idx'])
                constant_count_start_ns = file['events/t'][self.constant_count_length]
                constant_count_start_ms = constant_count_start_ns/1e6
                self.constant_count_start_ms = math.ceil(constant_count_start_ms/100)*100
            self.event_repr = StackedHistogram(10, self.image_h, self.image_w)
        else: # rgb and grayscale
            img_path = join(seq_path, 'imgs') 
            self.img_list = sorted(glob(img_path + '/*.png', recursive = True))
            assert len(self.img_list) == int(self.img_list[-1].split('/')[-1].split('.')[-2])+1

        # get label
        if self.img_type == 'RGB': 
            self.label_path = label_path
        else: # grayscale and constant count
            self.label_path = join(dataset_path, 'labels.h5')

            timestamps_path = join(self.seq_path, 'timestamps.txt')
            with open(timestamps_path) as f:
                txt_data = f.read()

            self.img_timestamps_ms = [round(float(i)*1e3) for i in txt_data.split()]
            ms_array = np.arange(0, self.img_timestamps_ms[-1]+1, 1)
            self.img_ms_2_idx = np.searchsorted(self.img_timestamps_ms, ms_array, side='left')

        action_folder = os.path.basename(os.path.normpath(self.seq_path))
        self.action_name = action_folder.split('.')[0].replace('_', ' ')

        with h5py.File(self.label_path, 'r') as h5:
            self.num_labels = len(h5[self.subject_name][self.action_name]['cam_view_{}'.format(self.cam_id)])

    def __open_h5(self):
        if self.img_type == 'constcount':
            self.h5_dict = {
                'movie': h5py.File(self.movie_path, 'r'),
                'label': h5py.File(self.label_path, 'r'),
            }
        else: 
            self.h5_dict = {
                'label': h5py.File(self.label_path, 'r'),
                        }

        self.h5_opened = True

    def __close_h5(self):

        for i, h5 in self.h5_dict.items():
                h5.close()

        self.h5_opened = False

    def __len__(self): # TODO FIX
        if self.img_type == 'constcount':
            seq_ms_len = min(self.mov_ms_len, self.num_labels*self.frame_interval)
            return int(np.floor(seq_ms_len - self.constant_count_start_ms)/100)
        else: 
            return self.num_labels

    def __getitem__(self, idx: int):
        self.__open_h5()

        labels = self.h5_dict['label'][self.subject_name][self.action_name]['cam_view_{}'.format(self.cam_id)]
        label_timestamps = self.h5_dict['label'][self.subject_name][self.action_name]['timestamps_ms']

        if self.img_type == 'grayscale':
            img, landmarks = self.get_gray_img(idx, labels, label_timestamps)
        elif self.img_type == 'RGB':            
            img, landmarks = self.get_rgb_img(idx,labels, label_timestamps)
        elif self.img_type == 'constcount':
            img, landmarks = self.get_constcount_img(idx, labels, label_timestamps)
        else:
            NotImplementedError

        mask = self.get_uv_mask(landmarks)
        landmarks_final = torch.from_numpy(landmarks)

        label_image = self.uv_to_image(landmarks, mask)
        label_final = torch.from_numpy(label_image)

        sample = {'image': img, 'landmarks': landmarks_final, 'label':label_final, 'subject_name':self.subject_name, 'action_name': self.action_name}

        self.__close_h5()
        return sample

    def get_gray_img(self, idx, labels, label_timestamps):
        img_ms = idx*self.frame_interval
        img_id = self.img_ms_2_idx[img_ms]
        img_path = self.img_list[img_id]
        assert img_ms == self.img_timestamps_ms[img_id]
        assert img_id == int(img_path.split('/')[-1].split('.')[0])
        img = read_image(img_path)

        landmarks = np.round(labels[idx]).astype(np.int32)
        landmarks_ms = label_timestamps[idx]
        assert img_ms == landmarks_ms
        return img, landmarks

    def get_rgb_img(self, idx, labels, label_timestamps):
        img_ms = idx*self.frame_interval
        img_path = self.img_list[idx]
        img = read_image(img_path)/255.0
        
        landmarks = np.round(labels[idx]).astype(np.int32)
        landmarks_ms = label_timestamps[idx]
        assert img_ms == landmarks_ms
        return img, landmarks

    def get_constcount_img(self, idx, labels, label_timestamps):
        ms_end = self.constant_count_start_ms + idx*100
        idx_end = int(self.h5_dict['movie']['ms_to_idx'][ms_end])
        assert idx_end >= self.constant_count_length - 5 # give 5 pixel error
        idx_start = idx_end - self.constant_count_length
        if idx_start < 0: 
            idx_start = 0
        assert idx_start >= 0

        events = {'x': torch.tensor(self.h5_dict['movie']['events/x'][idx_start:idx_end], dtype=torch.long),
            'y': torch.tensor(self.h5_dict['movie']['events/y'][idx_start:idx_end], dtype=torch.long),
            'p': torch.tensor(self.h5_dict['movie']['events/p'][idx_start:idx_end], dtype=torch.long),
            't': torch.tensor(self.h5_dict['movie']['events/t'][idx_start:idx_end], dtype=torch.long),}

        ev_repr = self.event_repr.construct(events['x'], events['y'], events['p'], events['t'])
        img = torch.cat((ev_repr[:,0], ev_repr[:,1]), dim=0)   

        # get label 
        landmarks_idx = int(ms_end/self.frame_interval)   
        landmarks_ms = label_timestamps[landmarks_idx]
        assert ms_end == landmarks_ms

        landmarks = np.round(labels[landmarks_idx]).astype(np.int32)
        return img, landmarks

    def get_uv_mask(self, sample):

        # mask is used to make sure that pixel positions are in frame range.
        mask = np.ones(sample.shape[1]).astype(np.int32)
        mask[np.isnan(sample[0])] = 0
        mask[np.isnan(sample[1])] = 0
        mask[sample[0] >= self.image_w] = 0 # x dim
        mask[sample[0] < 0] = 0
        mask[sample[1] >= self.image_h] = 0 # y dim
        mask[sample[1] < 0] = 0

        return mask

    def uv_to_image(self, pix_locs, mask):
        label_image = np.zeros((self.num_joints, self.image_h, self.image_w)) 

        for fmidx, zipd in enumerate(zip(pix_locs[1], pix_locs[0], mask)):
            if zipd[2] == 1:  # write joint position only when projection within frame boundaries
                label_image[fmidx, zipd[0], zipd[1]] = 1

        return label_image



