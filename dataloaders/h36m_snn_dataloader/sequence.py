import h5py
import torch
import numpy as np
import torchvision
import pdb
import math
import os

from tqdm import tqdm
from glob import glob
from pathlib import Path
from os.path import join
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from dataloaders.data_representations import StackedHistogram, VoxelGrid
from torchvision.io import read_image

torch.manual_seed(0)

class Movie(Dataset):

    # The h5 files have structure:
    #
    # events.h5 (under subject, action, and camera directory)
    # ├── events 
    # │   ├── p (polarity)
    # │   ├── t 
    # │   ├── x
    # │   └── y
    # ├── ms_to_idx (maps ms to location of ms in timestamps)
    # └── cam_view (holds a single value of the camera id: 0-3)

    # labels.h5 (holds all labels) 
    # └── Subject name (eg.'S1', 'S5')
    #     └── Action name (eg. 'Directions', 'SittingDown 2')
    #       ├── timestamps_ms (timestamp of label in ms)
    #       └── cam_view_{} (where {} is camera id)

    def __init__(self, dataset_path, subject_id, cam_id, seq_path, frame_interval, ev_repr, seq_len, bin_length_per_stack, step_size, image, img_type, img_path=None):

        self.subject_name = subject_id
        self.cam_id = cam_id
        self.seq_path = seq_path
        self.frame_interval = frame_interval
        self.label_path = join(dataset_path, 'labels.h5')
        self.movie_path = join(self.seq_path, 'events.h5')
        self.frame = image
        self.img_type = img_type
        self.img_path = img_path
        self.constant_count_length = 7500

        self.num_channels = 2
        self.output_seq_len = seq_len # sequence length in ms 
        self.step_size = step_size # how much to step to get the next sequence in ms
        self.label_step_size = int(self.step_size/self.frame_interval) # how much label should step for next sequence labels
        self.bin_length_per_stack = bin_length_per_stack # duration of events per stack of histogram 
        self.num_bins = int(self.output_seq_len/self.bin_length_per_stack) # count of bins per sequence

        self.label_rate = int(self.output_seq_len/self.frame_interval) # label rate in sequence, max allowed is every 20 ms 

        # assert self.step_size == 100, "Step size other than 100 are not supported at the moment."

        self.image_w, self.image_h, self.num_joints = 320, 256, 13

        if ev_repr == 'voxel_grid':
            self.event_repr = VoxelGrid(self.num_bins, self.image_h, self.image_w, normalize = True)
        elif ev_repr == 'stacked_hist':
            self.event_repr_cnn = StackedHistogram(10, self.image_h, self.image_w)
            self.event_repr = StackedHistogram(self.num_bins, self.image_h, self.image_w)
        else:
            NotImplementedError

        if self.frame:
            # Get image paths for sequence
            img_path = join(self.seq_path, 'imgs') 
            if self.img_type == 'RGB':
                img_path = join(self.img_path, self.seq_path.split('/')[-4], 'Videos', self.seq_path.split('/')[-2], 'imgs')
                assert Path(img_path).exists()
            self.img_list = sorted(glob(img_path + '/*.png', recursive = True))
            
            # Get timestamps of frames
            if self.img_type == 'grayscale':
                timestamps_path = join(self.seq_path, 'timestamps.txt')
                with open(timestamps_path) as f:
                    txt_data = f.read()

                # Get ms to idx for img timestamps
                self.img_timestamps_ms = [round(float(i)*1e3) for i in txt_data.split()]
                ms_array = np.arange(0, self.img_timestamps_ms[-1]+1, 1)
                self.img_ms_2_idx = np.searchsorted(self.img_timestamps_ms, ms_array, side='left')

        # get action name of sequence
        action_folder = os.path.basename(os.path.normpath(self.seq_path))
        self.action_name = action_folder.split('.')[0].replace('_', ' ')
        
        # Get number of frames by looking at number of labels
        with h5py.File(self.label_path, 'r') as h5:
            self.last_ms = h5[self.subject_name][self.action_name]['timestamps_ms'][-1]

        with h5py.File(self.movie_path, 'r') as h5:
            self.mov_ms_len = len(h5['ms_to_idx'])
            constant_count_start_ns = h5['events/t'][self.constant_count_length]
            constant_count_start_ms = constant_count_start_ns/1e6
            self.constant_count_start_ms = math.ceil(constant_count_start_ms/100)*100

        self.h5_opened = False

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
        seq_ms_len = min(self.mov_ms_len, self.last_ms)    
        if self.img_type == 'constcount' and self.frame:
            # return int(np.floor(seq_ms_len - self.constant_count_start_ms)/self.output_seq_len)
            return int(np.floor(seq_ms_len - self.constant_count_start_ms)/(self.output_seq_len - self.step_size)) - 1# step size 100ms with seq len 200ms 
        else:
            # return int((seq_ms_len - (self.output_seq_len - self.step_size))/(self.output_seq_len - self.step_size))
            return int(seq_ms_len/self.output_seq_len)

    def __getitem__(self, idx: int):
        if not self.h5_opened:
            self.__open_h5()
        
        if self.frame and self.img_type == 'constcount':
            # ms_start = int(idx*self.output_seq_len) + self.constant_count_start_ms
            # ms_end = int((idx + 1)*self.output_seq_len) + self.constant_count_start_ms
            ms_start = int(idx*self.step_size) + self.constant_count_start_ms  # step size 100ms with seq len 200ms 
            ms_end = ms_start + self.output_seq_len
        else: 
            # ms_start = int(idx*self.output_seq_len) 
            # ms_end = int((idx + 1)*self.output_seq_len)
            ms_start = int(idx*self.step_size) # step size 100ms with seq len 200ms 
            ms_end = ms_start + self.output_seq_len # step size 100ms with seq len 200ms 
        
        idx_start = int(self.h5_dict['movie']['ms_to_idx'][ms_start])
        idx_end = int(self.h5_dict['movie']['ms_to_idx'][ms_end])

        if idx_start == idx_end:
            idx_end = idx_end + 1
            num_events = 1
        else: 
            num_events = 0

        assert idx_start >= 0 
        assert idx_end >= 0 
        assert idx_end > idx_start

        assert self.h5_dict['movie']['cam_view'][()] == self.cam_id
        
        # get img data
        # print(self.frame, self.img_type)
        if self.frame:
            if self.img_type == 'grayscale':
                img, img_timestamp_ms = self.get_gray_img(ms_start)
            elif self.img_type == 'RGB':            
                img, img_timestamp_ms = self.get_rgb_img(idx)
            elif self.img_type == 'constcount':
                img, img_timestamp_ms = self.get_constcount_img(idx, idx_start)
            else:
                NotImplementedError

            assert ms_start == img_timestamp_ms
            assert ms_end == img_timestamp_ms + self.output_seq_len

        # get event data
        events = self.get_events(idx_start, idx_end, ms_start, ms_end, num_events)
        
        # get label
        landmarks, label = self.get_label(ms_start, ms_end, idx)

        if self.frame:
            sample = {'event_representation': events, 'image': img, 'landmarks': landmarks, 'label':label, 'subject_name':self.subject_name, 'action_name':self.action_name}
        else: 
            sample = {'event_representation': events, 'landmarks': landmarks, 'label':label, 'subject_name':self.subject_name, 'action_name':self.action_name}

        self.__close_h5()

        return sample

    def get_gray_img(self, img_ms):
        img_id = self.img_ms_2_idx[img_ms]
        img_path = self.img_list[img_id]
        assert img_ms == self.img_timestamps_ms[img_id]
        assert img_id == int(img_path.split('/')[-1].split('.')[0])
        img = read_image(img_path)

        img_timestamp = img_ms
        return img, img_timestamp

    def get_rgb_img(self, idx):
        # img_ms = idx*self.output_seq_len
        img_ms = idx*self.step_size # for different shifts than seq len
        img_id = int(img_ms/self.frame_interval)
        img_path = self.img_list[img_id]
        img = read_image(img_path)/255.0

        img_timestamp = img_ms
        return img, img_timestamp

    def get_constcount_img(self, idx, event_idx_start):
        # ms_end = self.constant_count_start_ms + idx*self.output_seq_len
        ms_end = self.constant_count_start_ms + idx*self.step_size # step size 100ms with seq len 200ms 
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

        ev_repr = self.event_repr_cnn.construct(events['x'], events['y'], events['p'], events['t'])
        img = torch.cat((ev_repr[:,0], ev_repr[:,1]), dim=0)   

        assert idx_end == event_idx_start
        img_timestamp = ms_end
        return img, img_timestamp

    def get_events(self, idx_start, idx_end, ms_start, ms_end, num_events=0):

        # round_upper_bool = np.ceil(self.h5_dict['movie']['events/t'][idx_end-1]/(self.output_seq_len*1e6))*self.output_seq_len == ms_end
        # round_lower_bool = np.floor(self.h5_dict['movie']['events/t'][idx_start]/(self.output_seq_len*1e6))*self.output_seq_len == ms_start

        # when seq len 200ms 
        round_upper_bool = np.ceil(self.h5_dict['movie']['events/t'][idx_end-1]/(self.step_size*1e6))*self.step_size == ms_end
        if not round_upper_bool:
            # check lower bound of next idx 
            round_upper_bool = np.floor(self.h5_dict['movie']['events/t'][idx_end]/(self.step_size*1e6))*self.step_size == ms_end 
            if np.ceil(self.h5_dict['movie']['events/t'][idx_end-1]/(self.step_size*1e6))*self.step_size + self.step_size == np.floor(self.h5_dict['movie']['events/t'][idx_end]/(self.step_size*1e6))*self.step_size - self.step_size:
                round_upper_bool = True
        
        round_lower_bool = np.floor(self.h5_dict['movie']['events/t'][idx_start]/(self.step_size*1e6))*self.step_size == ms_start
        if not round_lower_bool:
            # check upper bound of previous idx     
            round_lower_bool = np.ceil(self.h5_dict['movie']['events/t'][idx_start-1]/(self.step_size*1e6))*self.step_size == ms_start
            if np.floor(self.h5_dict['movie']['events/t'][idx_start]/(self.step_size*1e6))*self.step_size - self.step_size == np.ceil(self.h5_dict['movie']['events/t'][idx_start-1]/(self.step_size*1e6))*self.step_size + self.step_size:
                round_lower_bool = True

        if num_events and not round_lower_bool and not round_upper_bool: 
            ev_repr= torch.zeros(self.num_bins, self.num_channels, self.image_h, self.image_w)
        else: 
            assert round_upper_bool # round to nearest upper 100 ms
            assert round_lower_bool

            events = {'x': torch.tensor(self.h5_dict['movie']['events/x'][idx_start:idx_end], dtype=torch.long),
                'y': torch.tensor(self.h5_dict['movie']['events/y'][idx_start:idx_end], dtype=torch.long),
                'p': torch.tensor(self.h5_dict['movie']['events/p'][idx_start:idx_end], dtype=torch.long),
                't': torch.tensor(self.h5_dict['movie']['events/t'][idx_start:idx_end], dtype=torch.long),}

            ev_repr = self.event_repr.construct(events['x'], events['y'], events['p'], events['t'])

        return ev_repr

    def get_label(self, ms_start, ms_end, idx):
        if self.img_type == 'constcount' and self.frame:
            idx_start = int(ms_start/self.frame_interval) + 1 
            idx_end = int(idx_start + self.output_seq_len/self.frame_interval)
        else:
            idx_start = idx*self.label_step_size + 1 # ms
            # idx_start = idx*self.label_rate + 1 # ms
            idx_end = idx_start + self.label_rate # ms
        
        assert idx_start >= 0 
        assert idx_end >= 0 

        labels = self.h5_dict['label'][self.subject_name][self.action_name]['cam_view_{}'.format(self.cam_id)]
        label_timestamps = self.h5_dict['label'][self.subject_name][self.action_name]['timestamps_ms'][idx_start:idx_end]
        landmarks = np.round(labels[idx_start:idx_end]).astype(np.int32)
        
        assert landmarks.shape[0] == self.label_rate

        assert (ms_start + self.output_seq_len) == label_timestamps[-1]
        assert ms_end == label_timestamps[-1]
        assert ms_start + self.frame_interval == label_timestamps[0]

        mask = self.get_uv_mask(landmarks)
        landmarks_final = torch.from_numpy(landmarks)

        label_image = self.uv_to_image(landmarks_final, mask)
        label_final = torch.from_numpy(label_image)

        assert label_final.shape[0] == self.label_rate
        # print('LABEL ID', idx_start, idx_end)
        # print('LABEL TS', label_timestamps)

        return landmarks_final, label_final

    def get_uv_mask(self, sample):

        # mask is used to make sure that pixel positions are in frame range.
        u = sample[:,0]
        v = sample[:,1]

        mask = np.ones((u.shape)).astype(np.int32)
        mask[np.isnan(u)] = 0
        mask[np.isnan(v)] = 0
        mask[u >= self.image_w] = 0 # x dim
        mask[u < 0] = 0
        mask[v >= self.image_h] = 0 # y dim
        mask[v < 0] = 0

        return mask

    def uv_to_image(self, pix_locs, mask):
        label_image = np.zeros((self.label_rate, self.num_joints, self.image_h, self.image_w)) 

        for lb_idx in range(self.label_rate):
            for fmidx, zipd in enumerate(zip(pix_locs[lb_idx][1], pix_locs[lb_idx][0], mask[lb_idx])):
                if zipd[2] == 1:  # write joint position only when projection within frame boundaries
                    label_image[lb_idx, fmidx, zipd[0], zipd[1]] = 1

        return label_image

if __name__ == '__main__':
    from dataloaders.h36m_snn_dataloader.provider import get_dataset
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import torch
    import configparser

    config_path = '/home/asude/master_thesis/experiments/h36m_cnn_snn_200ms_50hz_constcount/hybrid_cnn_snn_config_h36m_constcount_200ms.ini'
    # config_path = '/home/asude/master_thesis/experiments/h36m_cnn_snn_200ms_50hz/hybrid_cnn_snn_config_h36m_rgb_200ms.ini'
    # config_path = '/home/asude/thesis_final/master_thesis/experiments/h36m_hybrid_cnn_snn_rgb/hybrid_cnn_snn_config_h36m.ini'
    # config_path = '/home/asude/thesis_final/master_thesis/configs/hybrid_cnn_snn_config_h36m.ini'
    configfile = configparser.ConfigParser()
    configfile.read(config_path) # todo take from args
    dataloader_config = configfile['dataloader_parameters']

    h36m = get_dataset(dataloader_config)
    train_set = h36m.get_train_dataset()
    test_set = h36m.get_test_dataset()

    # PREPARE DATA LOADER
    params = {'batch_size': 64,
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

    for i, sample_batched in tqdm(enumerate(train_set), total=train_set.__len__(), desc='Total batches: '):
        break
        # print(i)
        img = sample_batched['image']#.to(device)
        inputs = sample_batched['event_representation']#.to(device)
        labels = sample_batched['label']#.to(device)
        gt_landmarks = sample_batched['landmarks']#.to(device)
        # pdb.set_trace()

        # plt.figure()
        # plt.imshow(img[0].sum(dim=[0]).numpy(), alpha=0.6)
        # plt.imshow(inputs[0].sum(dim=[0,1]).numpy(), cmap='gray', alpha=0.5)
        # for k in range(13):
        #     plt.scatter(gt_landmarks[0,:,0,k].numpy(), gt_landmarks[0,:,1,k].numpy(), s=5, marker='.', c=np.arange(1,11,1), cmap='cool')
        # plt.savefig('constcount_200ms_50hz{}.png'.format(i))

        # if i == 20: 
        #     break


    for i, sample_batched in tqdm(enumerate(test_set), total=test_set.__len__(), desc='Total batches: '):
        # break
        # print(i)
        # img = sample_batched['image']
        inputs = sample_batched['event_representation']#.to(device)
        labels = sample_batched['label']#.to(device)
        gt_landmarks = sample_batched['landmarks']#.to(device)
        labels = sample_batched['label']#.to(device)
        # pdb.set_trace()

        # plt.figure()
        # plt.imshow(img[0,0].numpy(), cmap='gray', alpha=0.6)
        # plt.imshow(inputs[0].sum(dim=[0,1]).numpy(), cmap='gray', alpha=0.5)
        # for k in range(13):
        #     plt.scatter(gt_landmarks[0,:,0,k].numpy(), gt_landmarks[0,:,1,k].numpy(), s=5, marker='.', c=np.arange(0,5,1), cmap='cool')
        # plt.savefig('img_event_constant_count_labels{}.png'.format(i))

        # plt.figure()
        # plt.imshow(inputs[0].sum(dim=[0,1]).numpy(), cmap='gray', alpha=0.2)
        # for k in range(13):
        #     plt.scatter(gt_landmarks[0,:,0,k].numpy(), gt_landmarks[0,:,1,k].numpy(), s=5, marker='.', c=np.arange(0,5,1), cmap='cool')
        # plt.savefig('event_constant_count_labels{}.png'.format(i))



    # print(train_set.__len__())
