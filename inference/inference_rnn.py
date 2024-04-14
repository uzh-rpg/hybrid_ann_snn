'''
Run validation for pure RNN.
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

from pathlib import Path
from dataloaders.snn_dataloader.provider import get_dataset
from torch_utils.visualization import *
from models.rnn import E2VIDRecurrent

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

    model = E2VIDRecurrent(model_params_config).to(device)
    model.load_state_dict(torch.load(flags.pretrained_dict_path))

    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    num_bins = inference_config.getint('num_bins') 
    print('WARNING! Inference other than 100 Hz is not supported at the moment.')

    val_running_mpjpe = 0.0
    val_running_mpjpe_avg10 = 0.0
    val_mpjpe_list = np.zeros((num_bins))
    val_tot_batch_size = 0  
    spikes_percentage = {} 

    model.eval()
    with torch.no_grad(): # computational graphs are not created
        for val_batch_id, val_sample_batched in tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Total batches: '):
            
            val_inputs = val_sample_batched['event_representation'].to(device)
            val_labels = val_sample_batched['label'].to(device)
            val_gt_landmarks = val_sample_batched['landmarks'].to(device)

            val_sample_batch_size = val_inputs.shape[0]
            val_tot_batch_size += val_sample_batch_size

            # forward + loss
            val_mpjpe_avg10 = 0
            val_prev_states = None
            for i in range(num_bins):
                val_output, val_states = model(val_inputs[:, i].float(), prev_states=val_prev_states) 

                val_prev_states = val_states
            
                val_pred_landmarks = get_pred_landmarks(val_output)
                val_mpjpe_temp = calc_mpjpe(val_pred_landmarks, val_gt_landmarks[:, :, i])
                val_mpjpe_avg10 += val_mpjpe_temp/num_bins
                
                val_mpjpe_list[i] += val_mpjpe_temp*val_sample_batch_size
            
            val_mpjpe = val_mpjpe_temp
            val_running_mpjpe_avg10 += val_mpjpe_avg10*val_sample_batch_size
            val_running_mpjpe += val_mpjpe*val_sample_batch_size

with open(flags.txt_path, 'w') as f: 
    f.write('%s:%s\n' % ('val_running_mpjpe_avg10', val_running_mpjpe_avg10.item()/val_tot_batch_size))
    f.write('%s:%s\n' % ('val_running_mpjpe', val_running_mpjpe.item()/val_tot_batch_size))
    f.write('%s:%s\n' % ('val_mpjpe_list', val_mpjpe_list/val_tot_batch_size))
