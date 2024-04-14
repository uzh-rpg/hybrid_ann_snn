import re
import pdb
import h5py
import json

from pathlib import Path
from torch.utils.data import ConcatDataset
from pose_network_3d.dataloaders.pose_network_dataloader.sequence import Movie


'''
Note S16_session2_mov1 has been deleted from dataset due to non-sequential timestamps. 
It is however still in the constant count (ANN-based) dataset. 
'''

class get_dataset:
    # def __init__(self, kwargs):
    def __init__(self, dataset_path, receptive_field):

        self.dataset_path = dataset_path
        self.receptive_field = receptive_field

        self.sequence_list = list()

        p = Path(self.dataset_path)
        for child in p.glob('*_2D_3D.h5'):
            self.sequence_list.append(child)

    def get_train_dataset(self):
        train_sequence = list()

        for child in self.sequence_list:
            subject_id_str = child.stem.split('_')[0]
            subject_id = re.findall('[0-9]+', subject_id_str)[0]

            if int(subject_id) < 13:
                movie_path = child
                train_sequence.append(Movie(movie_path, self.receptive_field))
        
        return ConcatDataset(train_sequence)

    def get_test_dataset(self):
        test_sequence = list()
                    
        for child in self.sequence_list:
            subject_id_str = child.stem.split('_')[0]
            subject_id = re.findall('[0-9]+', subject_id_str)[0]
            
            if int(subject_id) >= 13:
                movie_path = child
                test_sequence.append(Movie(movie_path, self.receptive_field))
    
        return ConcatDataset(test_sequence)

if __name__ == '__main__':
    dataset_provider = get_dataset('/media/mathias/moar/asude/pose_network_3d/', 243)
    dhp19_dataset_train = dataset_provider.get_train_dataset()
    dhp19_dataset_test = dataset_provider.get_test_dataset()

    for i, samp in enumerate(dhp19_dataset_test):
        print(i)
    for i, samp in enumerate(dhp19_dataset_train):
        print(i)
        # pdb.set_trace()


