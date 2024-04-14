'''
Calculate score and loss for the case network doesn't learn to do anything to know baseline. 
'''
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from configs.args import FLAGS
from torch.optim import Adam
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, encoding

from pathlib import Path
from dataloaders.snn_dataloader.provider import get_dataset
from torch_utils.visualization import *
from models.hybrid_cnn_snn import Hybrid_CNN_SNN

if __name__ == '__main__':
    flags, configfile = FLAGS(inference=True)

    # get config
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
    model.load_state_dict(torch.load(flags.pretrained_dict_path))

    num_bins = inference_config.getint('num_bins') 

    val_running_mpjpe = 0.0
    val_running_mpjpe_avg10 = 0.0
    val_mpjpe_list = np.zeros((num_bins))
    val_tot_batch_size = 0  

    model.eval()
    with torch.no_grad(): # computational graphs are not created
        for val_batch_id, val_sample_batched in tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Total batches: '):
            
            val_inputs = val_sample_batched['event_representation'].to(device)
            val_labels = val_sample_batched['label'].to(device)
            val_gt_landmarks = val_sample_batched['landmarks'].to(device)
            val_inputs_cnn = val_sample_batched['cnn_input'].to(device)
            
            val_sample_batch_size = val_inputs.shape[0]
            val_tot_batch_size += val_sample_batch_size

            # forward + loss
            val_mpjpe_avg10 = 0
            for i in range(num_bins):
                if i == 0: 
                    _, val_cnn_output, mid_res = model(val_inputs[:,i].float(), val_inputs_cnn.float(), v_init = True)

                val_pred_landmarks = get_pred_landmarks(val_cnn_output)
                val_mpjpe_temp = calc_mpjpe(val_pred_landmarks, val_gt_landmarks[:, :, i])
                val_mpjpe_avg10 += val_mpjpe_temp/10

                val_mpjpe_list[i] += val_mpjpe_temp*val_sample_batch_size

            val_mpjpe = val_mpjpe_temp

            val_running_mpjpe_avg10 += val_mpjpe_avg10*val_sample_batch_size
            val_running_mpjpe += val_mpjpe*val_sample_batch_size
    
    with open(flags.txt_path, 'w') as f: 
        f.write('%s:%s\n' % ('val_running_mpjpe_avg10', val_running_mpjpe_avg10.item()/val_tot_batch_size))
        f.write('%s:%s\n' % ('val_running_mpjpe', val_running_mpjpe.item()/val_tot_batch_size))
        f.write('%s:%s\n' % ('val_mpjpe_list', val_mpjpe_list/val_tot_batch_size))
    