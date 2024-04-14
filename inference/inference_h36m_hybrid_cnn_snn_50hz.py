'''
Run validation for HYBRID CNN - SNN.
'''
import pdb
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from configs.args import FLAGS
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, encoding
from torch_utils.network_initialization import *
from pathlib import Path
from dataloaders.h36m_snn_dataloader.provider import get_dataset
from torch_utils.visualization import *
from models.hybrid_cnn_snn import Hybrid_CNN_SNN

if __name__ == '__main__':
    flags, configfile = FLAGS(inference=True)

    # get config
    model_init_config = configfile['model_initialization']
    model_params_config = configfile['model_parameters']
    dataloader_config = configfile['dataloader_parameters']
    inference_config = configfile['inference_parameters']

    # SET DEVICE
    device = torch.device(flags.device)
    print('Running on', device)

    # PREPARE DATA LOADER
    dhp19_dataset = get_dataset(dataloader_config) # todo
    val_set = dhp19_dataset.get_test_dataset() 

    val_loader = DataLoader(
        val_set,
        batch_size = inference_config.getint('batch_size'),
        shuffle = False,
        num_workers = inference_config.getint('num_workers'),
        pin_memory = True,
    )

    model = Hybrid_CNN_SNN(kwargs=model_params_config, inference=True).to(device)
    model.load_state_dict(torch.load(flags.pretrained_dict_path, map_location=device), strict=True)

    # all_snn true to get mid res block spikes, hybrid = True to hook correct spike names for hybrid model
    get_spikes = manage_hooks(model, hybrid = True, all_snn = True)

    num_bins = inference_config.getint('num_bins')
    num_labels = inference_config.getint('num_labels')
    print('WARNING! Inference other than 100 Hz is not supported at the moment.')

    val_running_mpjpe = 0.0
    val_running_mpjpe_avg = 0.0
    val_mpjpe_list = np.zeros((num_labels))
    val_tot_batch_size = 0  
    spikes_percentage = {} 

    model.eval()
    print('eval mode activated')

    with torch.no_grad(): # computational graphs are not created
        for val_batch_id, val_sample_batched in tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Total batches: '):
            
            val_inputs_cnn = val_sample_batched['image'].to(device)
            val_inputs = val_sample_batched['event_representation'].to(device)
            val_gt_landmarks = val_sample_batched['landmarks'].to(device)
            pdb.set_trace()

            val_sample_batch_size = val_inputs.shape[0]
            val_tot_batch_size += val_sample_batch_size

            # reset neurons
            functional.reset_net(model)

            # prepare hook for saving spikes
            activation = get_spikes.register_forward_hooks()


            # forward + loss
            val_mpjpe_avg = 0
            for i in range(num_bins):
                if i == 0: 
                    val_output, val_cnn_output, mid_res = model(val_inputs[:, i].float(), val_inputs_cnn.float(), v_init=True) # todo
                else:                   
                    val_output, mid_res = model(val_inputs[:, i].float(), v_init=False)
                val_pred_landmarks = get_pred_landmarks(val_output)
                val_mpjpe_temp = calc_mpjpe(val_pred_landmarks, val_gt_landmarks[:, i].float()).item()
                val_mpjpe_avg += val_mpjpe_temp/num_labels
                val_mpjpe_list[i] += val_mpjpe_temp*val_sample_batch_size
                
                if spikes_percentage == {}:
                    for key in activation.keys(): 
                        spikes_percentage[key] = np.zeros((num_bins))
                    spikes_percentage['mid_residualBlock'] = np.zeros((num_bins))

                for key in activation.keys(): 
                    # batch size, channel size, height, width
                    c, h, w = activation[key].shape[-3:]
                    spikes_out = activation[key]

                    spikes_percentage[key][i] += torch.sum(spikes_out).item()/(c*h*w)
                        
                spikes_percentage['mid_residualBlock'][i] += torch.sum(mid_res).item()/(256*32*32)

            # last mpjpe
            val_mpjpe = val_mpjpe_temp

            val_running_mpjpe_avg += val_mpjpe_avg*val_sample_batch_size
            val_running_mpjpe += val_mpjpe*val_sample_batch_size


    with open(flags.txt_path, 'w') as f: 
        for key, value in spikes_percentage.items(): 
            f.write('%s:%s\n' % (key, value/val_tot_batch_size))
        
        for key, value in spikes_percentage.items(): 
            f.write('total %s:%s\n' % (key, sum(value)/val_tot_batch_size))
        
        f.write('%s:%s\n' % ('val_running_mpjpe_avg10', val_running_mpjpe_avg/val_tot_batch_size))
        f.write('%s:%s\n' % ('val_running_mpjpe', val_running_mpjpe/val_tot_batch_size))
        f.write('%s:%s\n' % ('val_mpjpe_list', val_mpjpe_list/val_tot_batch_size))
    