import re
import pdb
import h5py
import json
import os

from glob import glob
from pathlib import Path
from torch.utils.data import ConcatDataset
from dataloaders.h36m_snn_dataloader.sequence import Movie


'''
Note __ALL__ sessions from all subjects have been removed because they do not exist in the labels provided. 
Additionally, S11 sequences 'Directions' have been removed due to corruption. 
'''

class get_dataset:
    def __init__(self, kwargs):

        self.dataset_path = kwargs['dataset_path']
        self.cam_id_list = json.loads(kwargs['camera_views'])
        self.image = kwargs.getboolean('image')
        self.img_type = kwargs['img_type']

        if self.img_type == 'RGB':
            self.img_path = kwargs['image_path']
        else: 
            self.img_path = None

        self.frame_interval = kwargs.getint('frame_interval') # ms # currently only supports 20 ms 
        self.subjects_train = ['S1',  'S5',  'S6',  'S7', 'S8']
        self.subjects_test = ['S9', 'S11']

        self.cam_id_map = {'54138969': 0, '55011271': 1, '58860488': 2, '60457274': 3}

        self.ev_repr = kwargs['ev_representation']
        self.seq_len = kwargs.getint('sequence_length')
        self.bin_length_per_stack = kwargs.getint('bin_length_per_stack')
        self.step_size = kwargs.getint('step_size')

    def get_train_dataset(self):
        train_sequence = list()
        for subject_id in self.subjects_train: 
            # if self.image and self.img_type == 'RGB':
            #     subject_path = os.path.join(self.dataset_path, subject_id, 'Videos')
            # else:
            subject_path = os.path.join(self.dataset_path, subject_id, 'upsample')

            seq_list = glob(subject_path + '/*/', recursive = True)
            for seq_id, seq_path in enumerate(seq_list):
                cam_code = seq_list[seq_id].split('/')[-2].split('.')[-1]
                cam_id = self.cam_id_map[cam_code]
                
                if cam_id in self.cam_id_list:
                    train_sequence.append(Movie(dataset_path=self.dataset_path, subject_id=subject_id, cam_id=cam_id, seq_path=seq_path, \
                    frame_interval=self.frame_interval, ev_repr=self.ev_repr, seq_len=self.seq_len, bin_length_per_stack=self.bin_length_per_stack, \
                    step_size=self.step_size, image=self.image, img_type=self.img_type, img_path=self.img_path))
        return ConcatDataset(train_sequence)

    def get_test_dataset(self):
        test_sequence = list()
        for subject_id in self.subjects_test: 
            # if self.image and self.img_type == 'RGB':
            #     subject_path = os.path.join(self.dataset_path, subject_id, 'Videos')
            # else:
            subject_path = os.path.join(self.dataset_path, subject_id, 'upsample')

            seq_list = glob(subject_path + '/*/', recursive = True)
            for seq_id, seq_path in enumerate(seq_list):
                cam_code = seq_list[seq_id].split('/')[-2].split('.')[-1]
                cam_id = self.cam_id_map[cam_code]
                
                if cam_id in self.cam_id_list:
                    test_sequence.append(Movie(dataset_path=self.dataset_path, subject_id=subject_id, cam_id=cam_id, seq_path=seq_path, \
                    frame_interval=self.frame_interval, ev_repr=self.ev_repr, seq_len=self.seq_len, bin_length_per_stack=self.bin_length_per_stack, \
                    step_size=self.step_size, image=self.image, img_type=self.img_type, img_path=self.img_path))
        return ConcatDataset(test_sequence)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torch

    h36m = get_dataset()
    train_set = h36m.get_train_dataset()
    test_set = h36m.get_test_dataset()

    # PREPARE DATA LOADER
    params = {'batch_size': 8,
        'num_workers': 4,
        'pin_memory': True}

    train_loader = DataLoader(
        train_set,
        shuffle = True,
        **params
    )
    # print(train_set.__len__())
    # print(test_set.__len__())

    for i, sample_batched in enumerate(train_set):
        inputs = sample_batched['image']
        labels = sample_batched['label']
        gt_landmarks = sample_batched['landmarks']
        print(i)
