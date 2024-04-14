import re
import pdb
import h5py
import json

from pathlib import Path
from torch.utils.data import ConcatDataset
from dataloaders.snn_dataloader.sequence import Movie


'''
Note S16_session2_mov1 has been deleted from dataset due to non-sequential timestamps. 
It is however still in the constant count (ANN-based) dataset. 
'''

class get_dataset:
    def __init__(self, kwargs):

        self.dataset_path = kwargs['dataset_path']
        self.P_mat_path = kwargs['p_mat_path']
        self.ev_repr = kwargs['ev_representation']

        # disabled funtionalities 
        # self.normalize = normalize
        # self.delta_preds = delta_preds

        self.cam_id_list = json.loads(kwargs['camera_views'])
        self.seq_len = kwargs.getint('sequence_length')
        self.cnn_input = kwargs.getboolean('cnn_input')
        self.constant_count = kwargs.getboolean('constant_count')
        self.constant_duration = kwargs.getboolean('constant_duration')
        self.bin_length_per_stack = kwargs.getint('bin_length_per_stack')
        self.step_size = kwargs.getint('step_size')

        self.label_path = self.dataset_path
        self.label_str = '_label.h5'

        self.sequence_list = list()

        p = Path(self.dataset_path)
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
                    train_sequence.append(Movie(movie_path, label_path, self.P_mat_path, self.ev_repr, self.bin_length_per_stack, self.step_size, cam_id, self.seq_len, self.cnn_input, self.constant_count, self.constant_duration))
        
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
                    test_sequence.append(Movie(movie_path, label_path, self.P_mat_path, self.ev_repr, self.bin_length_per_stack, self.step_size, cam_id, 200, self.cnn_input, self.constant_count, self.constant_duration))
        
        return ConcatDataset(test_sequence)

