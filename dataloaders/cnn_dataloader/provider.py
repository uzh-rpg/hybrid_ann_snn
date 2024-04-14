from dataloaders.cnn_dataloader.sequence import Movie
from torch.utils.data import ConcatDataset
from pathlib import Path

import re
import pdb
import h5py
import json

class get_dataset:
    def __init__(self, kwargs):

        self.dataset_path = kwargs['dataset_path']
        self.P_mat_path = kwargs['p_mat_path']
        self.ev_repr = kwargs['ev_representation']

        self.cam_id_list = json.loads(kwargs['camera_views'])
        self.step_size = kwargs.getint('step_size')
        
        self.constant_count = kwargs['constant_count']
        self.constant_duration = kwargs['constant_duration']

        self.seq_len = kwargs.getint('sequence_len')
        self.step_size = kwargs.getint('step_size') #ms # how much to step to get the next sequence
        self.num_bins = kwargs.getint('num_bins')

        assert self.constant_count != self.constant_duration

        self.label_path = self.dataset_path
        self.label_str = '_label.h5'

        self.sequence_list = list()

        p = Path(str(self.dataset_path))
        for child in p.glob('*events.h5'):
            self.sequence_list.append(child)

    def get_train_dataset(self):
        train_sequence = list()

        for cam_id in self.cam_id_list:

            for child in self.sequence_list:
                subject_id_str = child.stem.split('_')[0]
                subject_id = re.findall('[0-9]+', subject_id_str)[0]

                if int(subject_id) < 13:
                    movie_path = child
                    label_path = str(self.label_path) + child.stem + self.label_str
                    train_sequence.append(Movie(movie_path, label_path, self.P_mat_path, cam_id=cam_id, ev_repr = self.ev_repr, step_size=self.step_size, seq_len = self.seq_len, num_bins=self.num_bins, constant_count=self.constant_count, constant_duration=self.constant_duration))
        
        return ConcatDataset(train_sequence)

    def get_test_dataset(self):
        test_sequence = list()

        for cam_id in self.cam_id_list:

            for child in self.sequence_list:

                subject_id_str = child.stem.split('_')[0]
                subject_id = re.findall('[0-9]+', subject_id_str)[0]

                if int(subject_id) >= 13:
                    movie_path = child
                    label_path = str(self.label_path) + child.stem + self.label_str
                    test_sequence.append(Movie(movie_path, label_path, self.P_mat_path, cam_id=cam_id, ev_repr = self.ev_repr, step_size=self.step_size, seq_len = self.seq_len, num_bins=self.num_bins, constant_count=self.constant_count, constant_duration=self.constant_duration))

        return ConcatDataset(test_sequence)

if __name__ == '__main__':
    import configparser
    from torch.utils.data import DataLoader

    configfile = configparser.ConfigParser()
    configfile.read('/home/asude/master_thesis/configs/cnn_config.ini')

    dataloader_config = configfile['dataloader_parameters']

    # PREPARE DATA LOADER
    dhp19_dataset = get_dataset(dataloader_config) 
    train_set = dhp19_dataset.get_train_dataset() 
    val_set = dhp19_dataset.get_test_dataset() 

    params = {'batch_size': 8,
        'num_workers': 4,
        'pin_memory': True}

    train_loader = DataLoader(
        train_set,
        shuffle = True,
        **params
    )

    val_loader = DataLoader(
        val_set,
        shuffle = False,
        **params, 
    )

    for batch_id, sample_batched in enumerate(train_set):
        inputs = sample_batched['event_representation']
        labels = sample_batched['label']
        gt_landmarks = sample_batched['landmarks']
        pdb.set_trace()
