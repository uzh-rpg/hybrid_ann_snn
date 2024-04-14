import os 
import pdb
import h5py
import re 
import numpy as np 
import collections

subjects = ['S1', 'S11',  'S5',  'S6',  'S7', 'S8',  'S9']
cam_views = [0, 1, 2, 3]
label_dir = '/longtermdatasets/asude/human3.6m_downloader/training/data_2d_h36m_gt_13joints.npz'
new_label_dir = '/longtermdatasets/asude/h36m_events/labels.h5'
txt_path = '/home/asude/master_thesis/h36m_dataset_generate/crop_video_coordinates.txt'

def nested_dict():
    return collections.defaultdict(nested_dict)

with open(txt_path) as f:
    lines = f.readlines()

labels_handle = np.load(label_dir, allow_pickle=True)
raw_positions_2d = labels_handle['positions_2d'][()]


def read_txt_file(lines):

    crop_dict = nested_dict()

    for line in lines: 
        split_column = line.split(' : ')
        subject = split_column[0]

        split_arrow = split_column[1].split('   -->   ')
        split_view = split_arrow[0].split(' view ')
        action_name = split_view[0]

        # remove last space in action name  
        if not re.findall(r'\d+', action_name):
            action_name = action_name[:-1] 

        cam_view = int(split_view[1])
        split_arrow[1].split('x_start ')
        split_arrow[1].split('y_start ')

        # get int crop values
        x_crop_str, y_crop_str = re.findall(r'\d+',split_arrow[1])
        x_crop = int(x_crop_str)
        y_crop = int(y_crop_str)

        # fill dictionary
        crop_dict[subject][action_name][cam_view] = [x_crop, y_crop] 

    return crop_dict

crop_dict = read_txt_file(lines)

h5 = h5py.File(new_label_dir, 'w')

image_width = 1000
counter = 0
for subject in subjects: 
    subject_labels = raw_positions_2d[subject]
    for action in subject_labels.keys():
        action_labels = subject_labels[action]
        for cam_view in cam_views:
            labels = action_labels[cam_view]
            
            if cam_view == 3 or cam_view == 0: 
                image_height = 1002
            else: 
                image_height = 1000

            x_crop, y_crop = crop_dict[subject][action][cam_view]
            
            if x_crop > 40: 
                x_crop = 40

            if y_crop > (image_height - 768):
                y_crop = image_height - 768

            assert image_width - 960 >= x_crop >= 0
            assert image_height - 768 >= y_crop >=0

            # Adjust labels after rescale and crop
            labels[:,:,0] = (labels[:,:,0] - x_crop)/3
            labels[:,:,1] = (labels[:,:,1] - y_crop)/3

            labels_final = np.moveaxis(labels, 1, -1)

            h5_branch_name = subject + '/' + action + '/' + 'cam_view_' + str(cam_view) 
            print(h5_branch_name)
            h5.create_dataset(h5_branch_name, data=labels_final)
            counter += 1

            seq_len = labels.shape[0]

        h5_branch_name = subject + '/' + action + '/' + 'timestamps_ms'
        h5.create_dataset(h5_branch_name, data=np.arange(0, seq_len*20, 20))

h5.close()
print('Total # of sequences: ', counter)
