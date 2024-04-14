from dataloaders.h36m_cnn_dataloader.sequence import Movie
from torch.utils.data import ConcatDataset
from pathlib import Path
from glob import glob

import re
import pdb
import h5py
import json
import os

class get_dataset:
    def __init__(self, kwargs):

        self.dataset_path = kwargs['dataset_path']
        self.cam_id_list = json.loads(kwargs['camera_views'])

        self.frame_interval = 20 #ms # currently only supports 20 ms 
        self.subjects_train = ['S1',  'S5',  'S6',  'S7', 'S8']
        self.subjects_test = ['S9', 'S11']

        self.cam_id_map = {'54138969': 0, '55011271': 1, '58860488': 2, '60457274': 3}
        self.img_type = kwargs['img_type']
        if self.img_type == 'grayscale':
            self.label_path = None
        else: 
            self.label_path = kwargs['label_path']

    def get_train_dataset(self):
        train_sequence = list()
        for subject_id in self.subjects_train: 
            if self.img_type == 'RGB':
                subject_path = os.path.join(self.dataset_path, subject_id, 'Videos')
            else: # events
                subject_path = os.path.join(self.dataset_path, subject_id, 'upsample')

            seq_list = glob(subject_path + '/*/', recursive = True)
            for seq_id, seq_path in enumerate(seq_list):
                cam_code = seq_list[seq_id].split('/')[-2].split('.')[-1]
                cam_id = self.cam_id_map[cam_code]
                
                if cam_id in self.cam_id_list:
                    train_sequence.append(Movie(dataset_path=self.dataset_path, subject_id=subject_id, cam_id=cam_id, seq_path=seq_path, frame_interval=self.frame_interval, img_type=self.img_type, label_path=self.label_path))
        return ConcatDataset(train_sequence)

    def get_test_dataset(self):
        test_sequence = list()
        for subject_id in self.subjects_test: 
            if self.img_type == 'RGB':
                subject_path = os.path.join(self.dataset_path, subject_id, 'Videos')
            else: 
                subject_path = os.path.join(self.dataset_path, subject_id, 'upsample')

            seq_list = glob(subject_path + '/*/', recursive = True)
            for seq_id, seq_path in enumerate(seq_list):
                cam_code = seq_list[seq_id].split('/')[-2].split('.')[-1]
                cam_id = self.cam_id_map[cam_code]
                
                if cam_id in self.cam_id_list:
                    test_sequence.append(Movie(dataset_path=self.dataset_path, subject_id=subject_id, cam_id=cam_id, seq_path=seq_path, frame_interval=self.frame_interval, img_type=self.img_type, label_path=self.label_path))
        return ConcatDataset(test_sequence)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torch
    import configparser

    config_path = '/home/asude/thesis_final/master_thesis/experiments/h36m_constcount_camview0123/cnn_config_h36m_constcount.ini'
    configfile = configparser.ConfigParser()
    configfile.read(config_path) # todo take from args
    dataloader_config = configfile['dataloader_parameters']

    h36m = get_dataset(dataloader_config)
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

    val_loader = DataLoader(
        test_set,
        shuffle = True,
        **params
    )
    print(train_set.__len__())
    print(test_set.__len__())

    for i, sample_batched in enumerate(train_loader):
        inputs = sample_batched['image']
        labels = sample_batched['label']
        gt_landmarks = sample_batched['landmarks']

    for i, sample_batched in enumerate(val_loader):
        inputs = sample_batched['image']
        labels = sample_batched['label']
        gt_landmarks = sample_batched['landmarks']
        # action_name = sample_batched['action_name']
        # subject_name = sample_batched['subject_name']
    #     if subject_name == 'S9':
    #         # print(action_name)
    #         if action_name == 'Greeting' or action_name =='SittingDown 1' or action_name=='Waiting 1':
    #             counter += 1
    # print(counter)

        # plt.figure()
        # plt.imshow(inputs[0,0], cmap='gray')
        # plt.imshow(torch.sum(labels[0], dim=0), alpha = 0.5) # plot label
        # plt.scatter(gt_landmarks[0,0,:], gt_landmarks[0,1,:], color='r', s=5) # plot 2d labdmarks
        # plt.savefig('data_visualizations/constcount{}.png'.format(i))
        # if i == 20:
        #     break

    # for i, sample_batched in enumerate(test_set):
    #     inputs = sample_batched['event_representation']
    #     labels = sample_batched['label']
    #     gt_landmarks = sample_batched['landmarks']
    #     action_name = sample_batched['action_name']

    #     print(action_name)




