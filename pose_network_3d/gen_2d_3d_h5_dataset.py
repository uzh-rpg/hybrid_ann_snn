import pdb 
import torch
import numpy as np
import torch.nn as nn
import h5py

from tqdm import tqdm
from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloaders.snn_dataloader.provider import get_dataset

from configs.args import *
from torch_utils.utils import *
from torch_utils.network_initialization import *
from torch_utils.visualization import *
from models.hybrid_cnn_snn import Hybrid_CNN_SNN
from spikingjelly.clock_driven import functional

_seed_ = 2022
np.random.seed(_seed_)
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True    
torch.backends.cudnn.allow_tf32 = True

# read triangulation config file 
root_dir = '/media/mathias/moar/asude/pose_network_3d/'
flags, configfile = FLAGS(inference=True)

# get config
model_params_config = configfile['model_parameters']
dataloader_config = configfile['dataloader_parameters']
inference_config = configfile['inference_parameters']
pretrained_weights = configfile['pretrained_weights']

# SET DEVICE
device = torch.device(flags.device)
print('Running on', device)

# PREPARE DATA LOADER
dhp19_dataset = get_dataset(dataloader_config) 

# todo choose 
# for saving train set 
# train_set = dhp19_dataset.get_train_dataset() 
# for saving test set
train_set = dhp19_dataset.get_test_dataset()

model_cam3 = Hybrid_CNN_SNN(kwargs=model_params_config).to(device)
model_cam2 = Hybrid_CNN_SNN(kwargs=model_params_config).to(device)

dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

pretrained_dict_cam3 = torch.load(pretrained_weights['pretrained_weights_camview3'])
pretrained_dict_cam2 = torch.load(pretrained_weights['pretrained_weights_camview2'])

model_cam3.load_state_dict(pretrained_dict_cam3, strict=True)
model_cam2.load_state_dict(pretrained_dict_cam2, strict=True)

model_cam2.eval()
model_cam3.eval()
print('eval mode activated')

num_bins = inference_config.getint('num_bins') 

h_add = 2
w_add = 34

add_landmark_shift = torch.ones((13,2), device=device)
add_landmark_shift[:,0] = w_add
add_landmark_shift[:,1] = h_add

add_landmark_shift_broadcast = torch.broadcast_to(add_landmark_shift, (10,13,2))

train_len = int(train_set.__len__()/2)
print(train_set.__len__())
with torch.no_grad(): # computational graphs are not created

    # FOR CHECKING 
    running_mpjpe_cam3 = 0.0
    running_mpjpe_cam2 = 0.0
    running_mpjpe_avg10_cam3 = 0.0
    running_mpjpe_avg10_cam2 = 0.0

    mpjpe_list_cam3 = np.zeros((num_bins))
    mpjpe_list_cam2 = np.zeros((num_bins))

    cur_filename = ''
    counter = 0
    counter_10s = 0
    for batch_id, sample_batched in tqdm(enumerate(train_set), total=train_len, desc='Total samples: '):

        if batch_id == train_len:
            break

        inputs_cam3 = sample_batched['event_representation'].unsqueeze(dim=0).to(device)
        labels_cam3 = sample_batched['label'].unsqueeze(dim=0).to(device)
        gt_landmarks_cam3 = sample_batched['landmarks'].unsqueeze(dim=0).to(device)
        inputs_cnn_cam3 = sample_batched['cnn_input'].unsqueeze(dim=0).to(device)
        label_3d_cam3 = sample_batched['label_3d'].unsqueeze(dim=0).to(device) # todo
        
        filename_cam3 = sample_batched['filename']
        idx_cam3 = sample_batched['idx']

        inputs_cam2 = train_set[batch_id + train_len]['event_representation'].unsqueeze(dim=0).to(device)
        labels_cam2 = train_set[batch_id + train_len]['label'].unsqueeze(dim=0).to(device)
        gt_landmarks_cam2 = train_set[batch_id + train_len]['landmarks'].unsqueeze(dim=0).to(device)
        inputs_cnn_cam2 = train_set[batch_id + train_len]['cnn_input'].unsqueeze(dim=0).to(device) # todo
        label_3d_cam2 = train_set[batch_id + train_len]['label_3d'].unsqueeze(dim=0).to(device) # todo
        
        filename_cam2 = train_set[batch_id + train_len]['filename']
        idx_cam2 = train_set[batch_id + train_len]['idx']

        assert filename_cam2 == filename_cam3
        assert idx_cam2 == idx_cam3

        if label_3d_cam2.isnan().any() or label_3d_cam3.isnan().any():
            nan_vals = 1
            # print('nan val found')
        else: 
            assert (label_3d_cam2 == label_3d_cam2).all()

        if cur_filename !=  filename_cam3:  # bug: last h5 file is never saved 
            if batch_id != 0: 
                h5.create_dataset('estimates_2d/camview2', data=poses_2d_camview2.cpu())
                h5.create_dataset('estimates_2d/camview3', data=poses_2d_camview3.cpu())

                h5.create_dataset('gt_2d/camview2', data=poses_2d_gt_camview2.cpu())
                h5.create_dataset('gt_2d/camview3', data=poses_2d_gt_camview3.cpu())
                
                h5.create_dataset('gt_3d', data=poses_3d_gt.cpu())

                h5.close()

                print('Saved h5 file ', cur_filename[:-12])

            cur_filename = filename_cam3

            assert sample_batched['file_len'] == train_set[batch_id + train_len]['file_len']
            file_len = sample_batched['file_len']

            h5 = h5py.File(root_dir + cur_filename[:-12] + '2D_3D.h5', 'w')

            poses_2d_camview3 = torch.zeros((file_len*10, 13, 2), device=device)
            poses_2d_camview2 = torch.zeros((file_len*10, 13, 2), device=device)
            poses_3d_gt =  torch.zeros((file_len*10, 13, 3), device=device)
            poses_2d_gt_camview3 = torch.zeros((file_len*10, 13, 2), device=device)
            poses_2d_gt_camview2 = torch.zeros((file_len*10, 13, 2), device=device)

            counter = 0
            counter_10s = 0

        # reset neurons
        functional.reset_net(model_cam3)
        functional.reset_net(model_cam2)

        # forward + loss
        mpjpe_avg10_cam3 = 0
        mpjpe_avg10_cam2 = 0
        for i in range(num_bins):
            if i == 0:
                output_cam3, _ = model_cam3(inputs_cam3[:,i].float(), inputs_cnn_cam3.float(), v_init = True)
                output_cam2, _ = model_cam2(inputs_cam2[:,i].float(), inputs_cnn_cam2.float(), v_init = True)
            else:
                output_cam3 = model_cam3(inputs_cam3[:,i].float(), v_init = False)
                output_cam2 = model_cam2(inputs_cam2[:,i].float(), v_init = False)

            pred_landmarks_cam3 = get_pred_landmarks(output_cam3)
            pred_landmarks_cam2 = get_pred_landmarks(output_cam2)

            mpjpe_temp_cam3 = calc_mpjpe(pred_landmarks_cam3, gt_landmarks_cam3[:, :, i])
            mpjpe_temp_cam2 = calc_mpjpe(pred_landmarks_cam2, gt_landmarks_cam2[:, :, i])

            mpjpe_avg10_cam3 += mpjpe_temp_cam3/num_bins
            mpjpe_avg10_cam2 += mpjpe_temp_cam2/num_bins

            mpjpe_list_cam3[i] += mpjpe_temp_cam3
            mpjpe_list_cam2[i] += mpjpe_temp_cam2

            poses_2d_camview3[counter] = pred_landmarks_cam3[0].T + add_landmark_shift
            poses_2d_camview2[counter] = pred_landmarks_cam2[0].T + add_landmark_shift
            counter += 1
                
        poses_3d_gt[counter_10s*10:(counter_10s + 1)*10] = torch.transpose(label_3d_cam3[0], 1, 2)
        poses_2d_gt_camview3[counter_10s*10:(counter_10s + 1)*10] = gt_landmarks_cam3[0].permute(1, 2, 0) + add_landmark_shift_broadcast
        poses_2d_gt_camview2[counter_10s*10:(counter_10s + 1)*10] = gt_landmarks_cam2[0].permute(1, 2, 0) + add_landmark_shift_broadcast

        counter_10s += 1

        running_mpjpe_avg10_cam3 += mpjpe_avg10_cam3
        running_mpjpe_avg10_cam2 += mpjpe_avg10_cam2

        running_mpjpe_cam3 += mpjpe_temp_cam3
        running_mpjpe_cam2 += mpjpe_temp_cam2

    with open(flags.txt_path, 'w') as f:
        f.write('%s:%s\n' % ('running_mpjpe_avg10_cam3', running_mpjpe_avg10_cam3.item()/train_len))
        f.write('%s:%s\n' % ('running_mpjpe_avg10_cam2', running_mpjpe_avg10_cam2.item()/train_len))
        f.write('%s:%s\n' % ('running_mpjpe_cam3', running_mpjpe_cam3.item()/train_len))
        f.write('%s:%s\n' % ('running_mpjpe_cam2', running_mpjpe_cam2.item()/train_len))
        f.write('%s:%s\n' % ('mpjpe_list_cam3', mpjpe_list_cam3/train_len))
        f.write('%s:%s\n' % ('mpjpe_list_cam2', mpjpe_list_cam2/train_len))

