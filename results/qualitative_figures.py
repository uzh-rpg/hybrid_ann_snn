import numpy as np
import pdb
import torch.nn as nn
import os
import h5py
import matplotlib.gridspec as gridspec
import configparser

from os.path import join
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, encoding

from models.cnn import UNet # todo
from pathlib import Path
from dataloaders.snn_dataloader.provider import get_dataset
from torch_utils.visualization import *
from torch_utils.utils_3d import *

from models.hybrid_cnn_snn import Hybrid_CNN_SNN # todoconfigfile = configparser.ConfigParser()
configfile = configparser.ConfigParser()

# set dataloader
P_mat_dir = '/home/rslsync/Resilio Sync/DHP19/P_matrices/'
h5_events_path = '/media/mathias/moar/asude/qualitative_comparison_files/'

if not Path(P_mat_dir).exists():
    P_mat_dir = '/scratch/asude/P_matrices/'
    assert Path(P_mat_dir).exists()

if not Path(h5_events_path).exists():
    h5_events_path = '/scratch/asude/dhp19_h5_event_files/all_events/346x260/'
    assert Path(h5_events_path).exists()

# set device
device = 'cpu'

# get validation set
root_dir = '/home/asude/thesis_final/master_thesis/results/200hz/'
config_path = '/home/asude/thesis_final/master_thesis/results/200hz/config.ini'
# config_path = '/home/asude/thesis_final/master_thesis/results/qualitative_figures/config.ini'

configfile.read(config_path) # todo take from args
dataloader_config = configfile['dataloader_parameters']
model_config = configfile['model_parameters']

dhp19_dataset = get_dataset(dataloader_config) # todo
val_set = dhp19_dataset.get_test_dataset()

# dhp19_dataset_cam2 = get_dataset(h5_events_path, P_mat_dir, step_size = 10, cnn_input = True, cam_id_list=[2], seq_len=100, constant_count=False, constant_duration=True) # todo
# val_set_cam2 = dhp19_dataset.get_test_dataset()

pink = '#FF33D7'
green = '#0AF718'

cnn_params = '/home/asude/thesis/master_thesis_asude_aydin/DHP19/torch_impl/params_SNN_comp/model_params_Adam_unet_stackedhist_100ms_bin_10*2=20channels_256x256_endlabels_1camview_lr_lb-1=0.0001_EPOCH_6.pt' # camera view 3
hybrid_params = '/media/mathias/moar/asude/checkpoints/const_count/HYBRID_CNN_SNN_CONST_COUNT_10x10=100ms_tau3.0_output_decay0.8_camview3_2.pt'
# hybrid_params = '/media/mathias/moar/asude/checkpoints/const_count/HYBRID_CNN_SNN_CONST_COUNT_20x5=100ms_tau4.5_output_decay0.9_camview3_1.pt'
# hybrid_params = '/home/asude/thesis/master_thesis_asude_aydin/rpg_e2vid/params/useful_params_best_val_score/model_params__HYBRID_EXP6_Adam_10x10=100ms_lr=5e-05_256x256_2channels_BS2_tau3.0_vthr1.0_output_tau0.8_camview3_conv+bn+leakyrelu+conv_bn_3.pt'
# hybrid_params_cam2 = '/home/asude/thesis/master_thesis_asude_aydin/rpg_e2vid/params/useful_params_best_val_score/model_params__HYBRID_EXP6_Adam_10x10=100ms_lr=5e-05_256x256_2channels_BS2_tau3.0_vthr1.0_output_tau0.8_camview2_conv+bn+leakyrelu+conv+bn_1.pt'

cnn_model = UNet(model_config).to(device)
hybrid_model = Hybrid_CNN_SNN(model_config).to(device)
# hybrid_model_cam2 = Hybrid_CNN_SNN(kwargs=network_kwargs, feedback = False, surrogate_function=surrogate.ATan(), detach_reset=False, exp6=True, output_tau=output_tau).to(device)
# hybrid_model_cam2.load_state_dict(torch.load(hybrid_params_cam2))

cnn_model.load_state_dict(torch.load(cnn_params, map_location=device))
hybrid_model.load_state_dict(torch.load(hybrid_params, map_location=device))
# selected ids = 4200 or 5000 (right knee lift) + 1300 (right leg lift) + 3600 or 3150(star jump) +  100 (jump)

idx_list = [4200]

for idx in idx_list:
    ex = val_set[idx]

    save_img = root_dir + 'qualitative_2d_idx{}.png'.format(idx)

    inputs = ex['event_representation']
    cnn_input = ex['cnn_input']
    landmarks = ex['landmarks']

    cnn_output = cnn_model(cnn_input.unsqueeze(dim=0).float())
    cnn_landmarks = get_pred_landmarks(cnn_output)

    inputs_sum = torch.sum(inputs, dim=[0,1])
    img = norm_image_tensor(inputs_sum.unsqueeze(dim=0))

    # plt.figure(figsize = (1,3), gridspec_kw = {'wspace':0, 'hspace':0})

    fig, axs = plt.subplots(1,3, figsize = (15,5), gridspec_kw = {'wspace':0, 'hspace':0})

    # CNN AT 10 HZ
    axs[0].imshow(img[0], cmap='gray')
    axs[0].scatter(landmarks[0,:9,:], landmarks[1,:9,:], color=green, marker='o', s=10, linewidths=1)
    axs[0].scatter(landmarks[0,-1,:], landmarks[1,-1,:], color=green, marker='o', s=1, linewidths=10)

    axs[0].scatter(cnn_landmarks[0,0], cnn_landmarks[0,1], color=pink, marker ='o', s=3, linewidths=10)

    axs[0].axis('off')

    # CNN 100 HZ
    axs[1].imshow(img[0], cmap='gray')

    axs[1].scatter(landmarks[0,:9,:], landmarks[1,:9,:], color=green, marker='o', s=10, linewidths=1)
    axs[1].scatter(landmarks[0,-1,:], landmarks[1,-1,:], color=green, marker='o', s=3, linewidths=10)
    for i in range(20):
        k = idx + 1 + i
        ex = val_set[k]
        print(k)
        # pdb.set_trace()
        inputs = ex['cnn_input']
        output = cnn_model(inputs.unsqueeze(dim=0).float())
        landmarks_pred = get_pred_landmarks(output)
        if i == 19:
            axs[1].scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=3, linewidths=10)
        else:
            axs[1].scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=10, linewidths=1)


    axs[1].axis('off')

    axs[2].imshow(img[0], cmap='gray')
    axs[2].scatter(landmarks[0,:9,:], landmarks[1,:9,:], color=green, marker='o', s=10, linewidths=1)
    axs[2].scatter(landmarks[0,-1,:], landmarks[1,-1,:], color=green, marker='o', s=3, linewidths=10)

    ex = val_set[idx]
    print(idx)

    inputs_cnn = ex['cnn_input']
    inputs = ex['event_representation']

    for i in range(20):

        if i == 0: 
            output, cnn_output = hybrid_model(inputs[i].unsqueeze(dim=0).float(), inputs_cnn.unsqueeze(dim=0).float(), v_init=True) # todo
        else:
            output = hybrid_model(inputs[i].unsqueeze(dim=0).float(), v_init=False)
        
        landmarks_pred = get_pred_landmarks(output)

        if i == 19:
            axs[2].scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=3, linewidths=10)
        else:
            axs[2].scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=10, linewidths=1)

    # axs[2].scatter(cnn_landmarks[0,0], cnn_landmarks[0,1], color=green, marker ='+', s=100, linewidths=1.5)
    # axs[2].scatter(landmarks[0,0], landmarks[1,0], color=pink, marker=',', s=10, linewidths=1)
    axs[2].axis('off')

    plt.tight_layout()

    plt.savefig(save_img,dpi=400)
# exit()
pdb.set_trace()
# load projection matrices
P_mat_cam2 = torch.from_numpy(np.load(join(P_mat_dir,'P2.npy'))).to(device)
P_mat_cam3 = torch.from_numpy(np.load(join(P_mat_dir,'P3.npy'))).to(device)

cameras_pos = np.load(join(P_mat_dir,'camera_positions.npy'))
point0 = torch.from_numpy(np.stack((cameras_pos[1], cameras_pos[2]))).to(device)

h = 260
w = 344
h_add = 2
w_add = 34

idx_list = [5000, 3750, 2300, 500]

for idx in idx_list:
    save_img = '3d_qualitarive_idx{}.png'.format(idx)

    fig, axs = plt.subplots(1,1, figsize = (5,5), gridspec_kw = {'wspace':0, 'hspace':0})

    ex = val_set[idx]
    ex_cam2 = val_set[idx + 6089]

    print(idx)

    inputs_cnn = ex['cnn_input']
    inputs = ex['event_representation']
    gt_landmarks = ex['landmarks']

    inputs_cnn_cam2 = ex_cam2['cnn_input']
    inputs_cam2 = ex_cam2['event_representation']
    gt_landmarks_cam2 = ex_cam2['landmarks']


    label_3d = ex['label_3d']

    functional.reset_net(hybrid_model)
    functional.reset_net(hybrid_model_cam2)

    for i in range(10):

        if i == 0: 
            output, cnn_output = hybrid_model(inputs[i].unsqueeze(dim=0).float(), inputs_cnn.unsqueeze(dim=0).float(), v_init=True) # todo
            output_cam2, cnn_output_cam2 = hybrid_model_cam2(inputs_cam2[i].unsqueeze(dim=0).float(), inputs_cnn_cam2.unsqueeze(dim=0).float(), v_init=True) # todo
        else:
            output = hybrid_model(inputs[i].unsqueeze(dim=0).float(), v_init=False)
            output_cam2 = hybrid_model_cam2(inputs_cam2[i].unsqueeze(dim=0).float(), v_init=False)
        
    landmarks_pred_cam3 = get_pred_landmarks(output)
    landmarks_pred_cam2 = get_pred_landmarks(output_cam2)

    # pdb.set_trace()
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    axs[0].imshow(torch.sum(inputs_cam2, axis = [0,1]), cmap='gray')
    axs[0].scatter(landmarks_pred_cam2[0, 0,:], landmarks_pred_cam2[0, 1,:], color=pink, marker='o', s=10, linewidths=1)
    axs[0].scatter(gt_landmarks_cam2[0,-1,:], gt_landmarks_cam2[1,-1,:], color=green, marker='o', s=1, linewidths=10)
    axs[0].axis('off')

    axs[1].imshow(torch.sum(inputs, axis = [0,1]), cmap='gray')
    axs[1].scatter(landmarks_pred_cam3[0, 0,:], landmarks_pred_cam3[0, 1,:], color=pink, marker='o', s=10, linewidths=1)
    axs[1].scatter(gt_landmarks[0,-1,:], gt_landmarks[1,-1,:], color=green, marker='o', s=1, linewidths=10)
    axs[1].axis('off')
    plt.savefig('cam_views_2_3_idx{}.png'.format(idx))

    # SCORE CALCULATIONS
    pred_2d_cam3 = landmarks_pred_cam3[0]
    pred_2d_cam2 = landmarks_pred_cam2[0]
    # pred_2d_cam3 = gt_landmarks[:,-1]
    # pred_2d_cam2 = gt_landmarks_cam2[:,-1]
    # pdb.set_trace()

    pred_2d_cam3[0] = pred_2d_cam3[0] + w_add
    pred_2d_cam3[1] = h - (pred_2d_cam3[1] + h_add)
    pred_2d_cam2[0] = pred_2d_cam2[0] + w_add
    pred_2d_cam2[1] = h - (pred_2d_cam2[1] + h_add)

    xyz_cam3 = project_uv_xyz_cam(pred_2d_cam3.double(), P_mat_cam2, device).unsqueeze(dim=1)
    xyz_cam2 = project_uv_xyz_cam(pred_2d_cam2.double(), P_mat_cam3, device).unsqueeze(dim=1)
    
    point1 = torch.cat((xyz_cam3, xyz_cam2), dim=1)
    pred_3d = torch.zeros((13,3), device=device)
    for joint_idx in range(13):
        intersection = find_intersection(point0, point1[joint_idx], device)
        pred_3d[joint_idx] = intersection[0]

    fs = 13
    fig = plt.figure(figsize=(fs,fs))
    ax = Axes3D(fig)
    plotSingle3Dframe(ax, label_3d[-1].T, c='red') # gt
    plotSingle3Dframe(ax, pred_3d, c='blue') # pred
    # plt.tight_layout()
    plt.savefig(save_img)

    # break

    # plt.tight_layout()



