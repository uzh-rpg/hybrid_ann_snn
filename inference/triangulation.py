import numpy as np
import pdb
import torch.nn as nn
import os
import h5py

from tqdm import tqdm
from datetime import datetime
from configs.args import FLAGS
from os.path import join
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, encoding

from pose_network_3d.dataloaders.snn_dataloader.provider import get_dataset
from torch_utils.visualization import *
from torch_utils.utils_3d import *
from models.hybrid_cnn_snn import Hybrid_CNN_SNN # todo

if __name__ == '__main__':
    print('Pass an empty string to pretrained_dict as pretrained weights are passed in from the config file.')
    flags, configfile = FLAGS(inference=True)

    # get config
    model_params_config = configfile['model_parameters']
    dataloader_config = configfile['dataloader_parameters']
    inference_config = configfile['inference_parameters']
    pretrained_weights = configfile['pretrained_weights']

    # SET DEVICE
    device = torch.device(flags.device)
    print('Running on', device)

    # load projection matrices
    P_mat_dir = dataloader_config['p_mat_path']
    P_mat_cam2 = torch.from_numpy(np.load(join(P_mat_dir,'P2.npy'))).to(device)
    P_mat_cam3 = torch.from_numpy(np.load(join(P_mat_dir,'P3.npy'))).to(device)

    P_mats = [P_mat_cam2, P_mat_cam3]
    cameras_pos = np.load(join(P_mat_dir,'camera_positions.npy'))

    point0 = torch.from_numpy(np.stack((cameras_pos[1], cameras_pos[2]))).to(device)

    h = 260
    w = 344
    h_add = 2
    w_add = 34

    # PREPARE DATA LOADER
    dhp19_dataset = get_dataset(dataloader_config) 
    val_set = dhp19_dataset.get_test_dataset() 

    model_cam3 = Hybrid_CNN_SNN(kwargs=model_params_config).to(device)
    model_cam2 = Hybrid_CNN_SNN(kwargs=model_params_config).to(device)

    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # rename model weights
    pretrained_dict_cam3 = torch.load(pretrained_weights['pretrained_weights_camview3'])
    pretrained_dict_cam2 = torch.load(pretrained_weights['pretrained_weights_camview2'])

    model_cam3.load_state_dict(pretrained_dict_cam3, strict=True)
    model_cam2.load_state_dict(pretrained_dict_cam2, strict=True)

    model_cam2.eval()
    model_cam3.eval()
    print('eval mode activated')

    num_bins = inference_config.getint('num_bins') 
    nan_vals = 0
    with torch.no_grad(): # computational graphs are not created

        val_running_mpjpe_cam3 = 0.0
        val_running_mpjpe_cam2 = 0.0
        val_running_mpjpe_avg10_cam3 = 0.0
        val_running_mpjpe_avg10_cam2 = 0.0

        val_mpjpe_list_cam3 = np.zeros((num_bins))
        val_mpjpe_list_cam2 = np.zeros((num_bins))
        mpjpe_3d_list = np.zeros((num_bins))

        val_len = int(val_set.__len__()/2)
        print(val_set.__len__())
        for val_id, val_sample_batched in tqdm(enumerate(val_set), total=val_len, desc='Total samples: '):
            if val_id == val_len:
                break
            val_inputs_cam3 = val_sample_batched['event_representation'].unsqueeze(dim=0).to(device)
            val_labels_cam3 = val_sample_batched['label'].unsqueeze(dim=0).to(device)
            val_gt_landmarks_cam3 = val_sample_batched['landmarks'].unsqueeze(dim=0).to(device)
            val_inputs_cnn_cam3 = val_sample_batched['cnn_input'].unsqueeze(dim=0).to(device)
            val_label_3d_cam3 = val_sample_batched['label_3d'].unsqueeze(dim=0).to(device) # todo
            
            val_filename_cam3 = val_sample_batched['filename']
            val_idx_cam3 = val_sample_batched['idx']

            val_inputs_cam2 = val_set[val_id + val_len]['event_representation'].unsqueeze(dim=0).to(device)
            val_labels_cam2 = val_set[val_id + val_len]['label'].unsqueeze(dim=0).to(device)
            val_gt_landmarks_cam2 = val_set[val_id + val_len]['landmarks'].unsqueeze(dim=0).to(device)
            val_inputs_cnn_cam2 = val_set[val_id + val_len]['cnn_input'].unsqueeze(dim=0).to(device) # todo
            val_label_3d_cam2 = val_set[val_id + val_len]['label_3d'].unsqueeze(dim=0).to(device) # todo
            
            val_filename_cam2 = val_set[val_id + val_len]['filename']
            val_idx_cam2 = val_set[val_id + val_len]['idx']

            assert val_filename_cam2 == val_filename_cam3
            assert val_idx_cam2 == val_idx_cam3

            if val_label_3d_cam2.isnan().any() or val_label_3d_cam3.isnan().any():
                nan_vals += 1
                print('nan val found')
            else: 
                assert (val_label_3d_cam2 == val_label_3d_cam2).all()

            # reset neurons
            functional.reset_net(model_cam3)
            functional.reset_net(model_cam2)

            # forward + loss
            val_mpjpe_avg10_cam3 = 0
            val_mpjpe_avg10_cam2 = 0
            pred_2d_cam2 = []
            pred_2d_cam3 = []
            for i in range(num_bins):
                if i == 0:
                    val_output_cam3, _ = model_cam3(val_inputs_cam3[:,i].float(), val_inputs_cnn_cam3.float(), v_init = True)
                    val_output_cam2, _ = model_cam2(val_inputs_cam2[:,i].float(), val_inputs_cnn_cam2.float(), v_init = True)
                else:
                    val_output_cam3 = model_cam3(val_inputs_cam3[:,i].float(), v_init = False)
                    val_output_cam2 = model_cam2(val_inputs_cam2[:,i].float(), v_init = False)

                val_pred_landmarks_cam3 = get_pred_landmarks(val_output_cam3)
                val_pred_landmarks_cam2 = get_pred_landmarks(val_output_cam2)

                val_mpjpe_temp_cam3 = calc_mpjpe(val_pred_landmarks_cam3, val_gt_landmarks_cam3[:, :, i])
                val_mpjpe_temp_cam2 = calc_mpjpe(val_pred_landmarks_cam2, val_gt_landmarks_cam2[:, :, i])

                val_mpjpe_avg10_cam3 += val_mpjpe_temp_cam3/num_bins
                val_mpjpe_avg10_cam2 += val_mpjpe_temp_cam2/num_bins

                val_mpjpe_list_cam3[i] += val_mpjpe_temp_cam3
                val_mpjpe_list_cam2[i] += val_mpjpe_temp_cam2

                # SCORE CALCULATIONS
                pred_2d_cam3 = val_pred_landmarks_cam3[0]
                pred_2d_cam2 = val_pred_landmarks_cam2[0]

                pred_2d_cam3[0] = pred_2d_cam3[0] + w_add
                pred_2d_cam3[1] = h - (pred_2d_cam3[1] + h_add)
                pred_2d_cam2[0] = pred_2d_cam2[0] + w_add
                pred_2d_cam2[1] = h - (pred_2d_cam2[1] + h_add)

                xyz_cam3 = project_uv_xyz_cam(pred_2d_cam3.double(), P_mat_cam2, device)
                xyz_cam2 = project_uv_xyz_cam(pred_2d_cam2.double(), P_mat_cam3, device)

                label_3d = val_label_3d_cam2
                pred_3d = torch.zeros((13,3), device=device)

                xyz_cam3 = xyz_cam3.unsqueeze(dim=1)
                xyz_cam2 = xyz_cam2.unsqueeze(dim=1)

                point1 = torch.cat((xyz_cam3, xyz_cam2), dim=1)
                for joint_idx in range(13):
                    intersection = find_intersection(point0, point1[joint_idx], device)
                    pred_3d[joint_idx] = intersection[0]

                mpjpe_3d_joints = torch.linalg.norm((label_3d[0,0].T - pred_3d), axis=-1)
                mpjpe_3d_sample = torch.nanmean(mpjpe_3d_joints)
                mpjpe_3d_list[i] += mpjpe_3d_sample

            val_running_mpjpe_avg10_cam3 += val_mpjpe_avg10_cam3
            val_running_mpjpe_avg10_cam2 += val_mpjpe_avg10_cam2

            val_running_mpjpe_cam3 += val_mpjpe_temp_cam3
            val_running_mpjpe_cam2 += val_mpjpe_temp_cam2

    print('total 3d label nan values found: ', nan_vals)
    with open(flags.txt_path, 'w') as f:
        f.write('%s:%s\n' % ('val_running_mpjpe_avg10_cam3', val_running_mpjpe_avg10_cam3.item()/val_len))
        f.write('%s:%s\n' % ('val_running_mpjpe_avg10_cam2', val_running_mpjpe_avg10_cam2.item()/val_len))
        f.write('%s:%s\n' % ('val_running_mpjpe_cam3', val_running_mpjpe_cam3.item()/val_len))
        f.write('%s:%s\n' % ('val_running_mpjpe_cam2', val_running_mpjpe_cam2.item()/val_len))
        f.write('%s:%s\n' % ('val_mpjpe_list_cam3', val_mpjpe_list_cam3/val_len))
        f.write('%s:%s\n' % ('val_mpjpe_list_cam2', val_mpjpe_list_cam2/val_len))
        f.write('%s:%s\n' % ('mpjpe_3d_list', mpjpe_3d_list/val_len))

