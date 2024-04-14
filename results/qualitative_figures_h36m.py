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
from dataloaders.h36m_snn_dataloader.provider import get_dataset
from torch_utils.visualization import *
from torch_utils.utils_3d import *

from models.hybrid_cnn_snn import Hybrid_CNN_SNN # todoconfigfile = configparser.ConfigParser()
root_dir = '/home/asude/thesis_final/master_thesis/results/'

configfile = configparser.ConfigParser()
config_path = '/home/asude/thesis_final/master_thesis/results/h36m_hybrid_rgb_init_rgb.ini'

# set device
device = 'cpu'

configfile.read(config_path) # todo take from args
dataloader_config = configfile['dataloader_parameters']
model_config = configfile['model_parameters']

dhp19_dataset = get_dataset(dataloader_config) # todo
val_set = dhp19_dataset.get_test_dataset()

pink = '#FF33D7'
green = '#0AF718'

# cnn_params = '/home/asude/thesis_final/master_thesis/experiments/checkpoints/CNN/2023-02-15_19-19-53_CNN_H36M_lr0.0001_BS32_RGB_camview0123_initlastweightdiv4000_divideinput255_7.pt' # all cam views 
hybrid_params = '/home/asude/thesis_final/master_thesis/experiments_200ms_100hz/rgb/2023-03-04_22-35-20_HYBRID_CNN_SNN_H36M_100Hz_lr5e-05_BS2_tau5.0_thr1.0_outputdecay0.9_200ms_all_views_init_hybrid_rgb_exp_shift100_epoch2_iter39999.pt'

# cnn_model = UNet(model_config).to(device)
hybrid_model = Hybrid_CNN_SNN(model_config).to(device)

# cnn_model.load_state_dict(torch.load(cnn_params, map_location=device))
hybrid_model.load_state_dict(torch.load(hybrid_params, map_location=device))

pdb.set_trace()
idx_list = np.arange(200,300,1)
# idx_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for idx in idx_list:
    ex = val_set[idx]

    save_img = root_dir + 'qualitative_2d_idx{}.png'.format(idx)

    inputs = ex['event_representation']
    cnn_input = ex['image']
    landmarks = ex['landmarks']
    # cnn_output = cnn_model(cnn_input.unsqueeze(dim=0).float())
    # cnn_landmarks = get_pred_landmarks(cnn_output)

    inputs_sum = torch.sum(inputs, dim=[0,1])
    # img = norm_image_tensor(inputs_sum.unsqueeze(dim=0))

    # plt.figure(figsize = (1,3), gridspec_kw = {'wspace':0, 'hspace':0})

    fig, axs = plt.subplots(1,1, figsize = (5,5), gridspec_kw = {'wspace':0, 'hspace':0})

    # # CNN AT 10 HZ
    # axs[0].imshow(img[0], cmap='gray')
    # axs[0].scatter(landmarks[0,:9,:], landmarks[1,:9,:], color=green, marker='o', s=10, linewidths=1)
    # axs[0].scatter(landmarks[0,-1,:], landmarks[1,-1,:], color=green, marker='o', s=1, linewidths=10)

    # axs[0].scatter(cnn_landmarks[0,0], cnn_landmarks[0,1], color=pink, marker ='o', s=3, linewidths=10)

    # axs[0].axis('off')

    # # CNN 100 HZ
    # axs[1].imshow(img[0], cmap='gray')

    # axs[1].scatter(landmarks[0,:9,:], landmarks[1,:9,:], color=green, marker='o', s=10, linewidths=1)
    # axs[1].scatter(landmarks[0,-1,:], landmarks[1,-1,:], color=green, marker='o', s=3, linewidths=10)
    # for i in range(20):
    #     k = idx + 1 + i
    #     ex = val_set[k]
    #     print(k)
    #     # pdb.set_trace()
    #     inputs = ex['cnn_input']
    #     output = cnn_model(inputs.unsqueeze(dim=0).float())
    #     landmarks_pred = get_pred_landmarks(output)
    #     if i == 19:
    #         axs[1].scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=3, linewidths=10)
    #     else:
    #         axs[1].scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=10, linewidths=1)


    # axs[1].axis('off')

    axs.imshow(inputs_sum, cmap='gray')
    axs.scatter(landmarks[:9,0,:], landmarks[:9,1,:], color=green, marker='o', s=6, linewidths=1)
    axs.scatter(landmarks[-1,0,:], landmarks[-1,1,:], color=green, marker='o', s=3, linewidths=10)

    # ex = val_set[idx]
    # print(idx)

    # cnn_input = ex['image']
    # inputs = ex['event_representation']

    for i in range(10):

        if i == 0: 
            output, cnn_output = hybrid_model(inputs[i].unsqueeze(dim=0).float(), cnn_input.unsqueeze(dim=0).float(), v_init=True) # todo
        else:
            output = hybrid_model(inputs[i].unsqueeze(dim=0).float(), v_init=False)
        
        landmarks_pred = get_pred_landmarks(output)

        if i == 9:
            axs.scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=3, linewidths=10)
        else:
            axs.scatter(landmarks_pred[0,0], landmarks_pred[0,1], color=pink, marker ='o', s=6, linewidths=1)

    # axs[2].scatter(cnn_landmarks[0,0], cnn_landmarks[0,1], color=green, marker ='+', s=100, linewidths=1.5)
    # axs[2].scatter(landmarks[0,0], landmarks[1,0], color=pink, marker=',', s=10, linewidths=1)
    axs.axis('off')

    plt.tight_layout()

    plt.savefig(save_img,dpi=400)
