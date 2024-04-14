from results.iccv.core import * 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import gcf

import numpy as np
import pdb
import torch.nn as nn
import os
import h5py
import configparser

from os.path import join
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, encoding

from pathlib import Path
from dataloaders.snn_dataloader.provider import get_dataset
from torch_utils.visualization import *
from torch_utils.utils_3d import *

from models.cnn import UNet # todo
from models.hybrid_cnn_snn import Hybrid_CNN_SNN 
from models.snn import EVSNN_LIF_final
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

configfile.read(config_path) # todo take from args
dataloader_config = configfile['dataloader_parameters']
model_config = configfile['model_parameters']

dhp19_dataset = get_dataset(dataloader_config) # todo
val_set = dhp19_dataset.get_test_dataset()

cnn_params = '/home/asude/thesis/master_thesis_asude_aydin/DHP19/torch_impl/params_SNN_comp/model_params_Adam_unet_stackedhist_100ms_bin_10*2=20channels_256x256_endlabels_1camview_lr_lb-1=0.0001_EPOCH_6.pt' # camera view 3
hybrid_params = '/media/mathias/moar/asude/checkpoints/const_count/HYBRID_CNN_SNN_CONST_COUNT_10x10=100ms_tau3.0_output_decay0.8_camview3_2.pt'

pink = '#FF33D7'
green = '#0AF718'

cnn_model = UNet(model_config).to(device)
hybrid_model = Hybrid_CNN_SNN(model_config, inference=True).to(device)

cnn_model.load_state_dict(torch.load(cnn_params, map_location=device))
hybrid_model.load_state_dict(torch.load(hybrid_params, map_location=device))

# idx_list = [4200, 4210, 4220, 4230]
# idx_list = [4200]
idx_list = [470, 480, 490, 500]
# idx_list = [470]
tot_frames = 81

all_hybrid_inputs = []
all_cnn_inputs = []

hybrid_landmarks_pred = []
cnn_landmarks_pred = []

all_cnn_landmarks = []
all_hybrid_landmarks = []

hybrid_mpjpe = []
cnn_mpjpe = []


num_mops_hybrid = {'static_conv_snn': 104.86,
'down1_snn': 838.86,
'down2_snn': 838.86,
'down3_snn': 838.86,
'mid_residualBlock': 603.98,
'residualBlock_snn': 603.98,
'up1_snn': 6710.89,
'up2_snn': 6710.89,
'up3_snn': 6710.89}

energy_snn = []
energy_hybrid = []
layer_names_hybrid = ['static_conv_snn', 'down1_snn', 'down2_snn', 'down3_snn', 'residualBlock_snn', 'up1_snn', 'up2_snn', 'up3_snn', 'mid_residualBlock']
get_hybrid_spikes = manage_hooks(hybrid_model, hybrid = True, all_snn = True)
for idx in idx_list: 

    # get sample
    ex = val_set[idx]

    inputs = ex['event_representation']
    cnn_inputs = ex['cnn_input']
    landmarks = ex['landmarks']
    cnn_landmarks = ex['cnn_landmarks']

    spikes_percentage = {}
    activation = get_hybrid_spikes.register_forward_hooks()
    for i in range(20):
        if i == 0: 
            output, cnn_output, mid_res = hybrid_model(inputs[i].unsqueeze(dim=0).float(), cnn_inputs[0].unsqueeze(dim=0).float(), v_init=True) # todo
            
            # get the very 1st prediction
            if idx == idx_list[0]:
                # pdb.set_trace()
                cnn_input_0 = torch.sum(cnn_inputs[0], dim=0)
                cnn_pred_0 = get_pred_landmarks(cnn_output)
                cnn_mpjpe_0 = calc_mpjpe(cnn_pred_0, cnn_landmarks[:,0].unsqueeze(dim=0))
                cnn_landmark_0 = cnn_landmarks[:,0]
        
        else:
            output, mid_res = hybrid_model(inputs[i].unsqueeze(dim=0).float(), v_init=False)
        
        all_hybrid_inputs.append(torch.sum(inputs[i], dim=0))
        pred_hybrid = get_pred_landmarks(output)
        hybrid_landmarks_pred.append(pred_hybrid)
        if i%2: 
            all_hybrid_landmarks.append(landmarks[:,int(i/2)])
            hybrid_mpjpe.append(calc_mpjpe(pred_hybrid, landmarks[:,int(i/2)].unsqueeze(dim=0)))
        
        if spikes_percentage == {}:
            for key in activation.keys(): 
                spikes_percentage[key] = np.zeros((20))
            spikes_percentage['mid_residualBlock'] = np.zeros((20))

        for key in activation.keys(): 
            # batch size, channel size, height, width
            c, h, w = activation[key].shape[-3:]
            spikes_out = activation[key]

            spikes_percentage[key][i] += torch.sum(spikes_out).item()/(c*h*w)
                
        spikes_percentage['mid_residualBlock'][i] += torch.sum(mid_res).item()/(256*32*32)

    for time in range(20):
        energy_per_timestep = 0
        for layer_name in layer_names_hybrid: 
            energy_per_timestep += spikes_percentage[layer_name][time]*num_mops_hybrid[layer_name]*0.38/1e3 # over time
        energy_hybrid.append(energy_per_timestep)

    for i in range(20):
        cnn_output = cnn_model(cnn_inputs[i+1].unsqueeze(dim=0).float())
        
        all_cnn_inputs.append(torch.sum(cnn_inputs[i+1], dim=0))
        pred_cnn = get_pred_landmarks(cnn_output)
        cnn_landmarks_pred.append(pred_cnn)
        if i%2: 
            all_cnn_landmarks.append(cnn_landmarks[:,int(i/2)+1])
            cnn_mpjpe.append(calc_mpjpe(pred_cnn, cnn_landmarks[:,int(i/2)+1].unsqueeze(dim=0)))

    # check correctness
    for i in range(10): 
        assert (cnn_landmarks[:,i+1] == landmarks[:,i]).all()

plt.rcParams.update({'font.family':'serif',
                     'font.size': 20})

for i in range(len(energy_hybrid)): 
    if not (i+1)%20: 
        energy_hybrid[i] += 37.175 + 2.155 # ann and state init energy
energy_hybrid_end = np.concatenate(([37.175 + 2.155], energy_hybrid))
energy_cnn = np.repeat(37.175, 81)

# pdb.set_trace()
input_time = []
label_time = []

y_energy_hybrid = []
y_energy_cnn = []

y_mpjpe_hybrid = []
y_mpjpe_cnn = []

def animate(i):
    print(i)
    
    input_time.append(timebins[i])
    y_energy_hybrid.append(energy_hybrid_end[i]/0.05)
    y_energy_cnn.append(energy_cnn[i]/0.05)
    if not i%2:
        label_time.append(label_bins[int(i/2)])
        if i ==0: 
            y_mpjpe_hybrid.append(cnn_mpjpe_0)
            y_mpjpe_cnn.append(cnn_mpjpe_0)
        else:
            y_mpjpe_hybrid.append(hybrid_mpjpe[int(i/2)-1])
            y_mpjpe_cnn.append(cnn_mpjpe[int(i/2)-1])
        

    ax0.clear()
    ax0.axis('off')
    if i == 0:
        ax0.imshow(all_hybrid_inputs[0], cmap='gray')
        ax0.scatter(cnn_landmark_0[0][0], cnn_landmark_0[0][1], color=green, marker ='+', s=100, linewidths=1.5)
        ax0.scatter(cnn_pred_0[0][0], cnn_pred_0[0][1], color=pink, marker=',', s=10, linewidths=1)
    else: 
        ax0.imshow(all_hybrid_inputs[i-1], cmap='gray')
        ax0.scatter(all_hybrid_landmarks[int(i/2)-1][0], all_hybrid_landmarks[int(i/2)-1][1], color=green, marker ='+', s=100, linewidths=1.5)
        ax0.scatter(hybrid_landmarks_pred[i-1][0,0], hybrid_landmarks_pred[i-1][0,1], color=pink, marker=',', s=10, linewidths=1)
    ax0.text(120, 250, 'Groundtruth', color=green, fontsize=20)
    ax0.text(140, 230, 'Estimation', color=pink, fontsize=20)
    ax0.set_title('Hybrid (Ours)', rotation='vertical',x=-0.1,y=0.2)
    
    # ax0.plot([0.36, 0.36], [0.15, 0.85], color='black', linestyle = '--', lw=3,
    #      transform=gcf().transFigure, clip_on=False)


    ax1.clear()
    if i == 0:
        ax1.imshow(cnn_input_0, cmap='gray')
        ax1.scatter(cnn_landmark_0[0][0], cnn_landmark_0[0][1], color=green, marker ='+', s=100, linewidths=1.5)
        ax1.scatter(cnn_pred_0[0][0], cnn_pred_0[0][1], color=pink, marker=',', s=10, linewidths=1)
    else:         
        ax1.imshow(all_cnn_inputs[i-1], cmap='gray')
        ax1.scatter(all_cnn_landmarks[int(i/2)-1][0], all_cnn_landmarks[int(i/2)-1][1], color=green, marker ='+', s=100, linewidths=1.5)
        ax1.scatter(cnn_landmarks_pred[i-1][0,0], cnn_landmarks_pred[i-1][0,1], color=pink, marker=',', s=10, linewidths=1)
    ax1.axis('off')
    ax1.set_title('ANN', rotation='vertical',x=-0.1,y=0.4)

    ax2.clear()
    ax2.plot(input_time, y_energy_hybrid, linewidth=2,  marker = '.', markersize=10, color='red', label='Hybrid (Ours)')
    ax2.plot(input_time, y_energy_cnn, linewidth=2, marker = '.', markersize=10, color='blue', label='ANN')
    ax2.set_xlim(0,400)
    # ax2.set_anchor((0, 10))

    ax2.legend(loc=(0.35,1.04), ncol=2)
    ax2.set_yscale('log')
    ax2.set_yticks([1, 10, 100, 1000])
    ax2.set_ylabel('Power (mW)')
    ax2.set_xlabel('Time [ms]')

    if not i%2:
        ax3.clear()
        ax3.plot(label_time, y_mpjpe_hybrid, linewidth=2, marker = '.', markersize=10, color='red', label='Hybrid (Ours)')
        ax3.plot(label_time, y_mpjpe_cnn, linewidth=2, marker = '.', markersize=10, color='blue', label='ANN')
    ax3.set_xlim(0,400)
    # ax3.set_ylim(1.5, 4.0) # for right leg lifts
    ax3.set_ylim(2.4, 5.0) # for right leg lifts
    ax3.set_ylabel('MPJPE (2D)')
    ax3.set_xlabel('Time [ms]')

# create grid for different subplots
fig = plt.figure(figsize=(16, 9), dpi=80)
fig.suptitle("Experiments at 200 Hz", fontweight="bold", fontsize=20)
 
timebins = np.arange(0,410,5)
label_bins = np.arange(0,410,10)

# axsnn = plt.subplot2grid(shape=(7, 9), loc=(0, 0), colspan=3, rowspan=3)
ax0 = plt.subplot2grid(shape=(10, 16), loc=(0, 0), colspan=5, rowspan=5)
# ax0.set_anchor((0,1))
ax1 = plt.subplot2grid(shape=(10, 16), loc=(5, 0), colspan=5, rowspan=5)
ax2 = plt.subplot2grid(shape=(8, 16), loc=(0, 6), colspan=11, rowspan=3)
ax3 = plt.subplot2grid(shape=(8, 16), loc=(4, 6), colspan=11, rowspan=3)
plt.subplots_adjust(wspace=3, hspace=0.1)

ani = FuncAnimation(fig, animate, frames=tot_frames, interval=200, repeat=False, blit=False)

plt.show()
ani.save('test4.mp4')
