'''
Run inference for pure CNN.
'''
import pdb
import torch.nn as nn
import torch
import numpy as np 

from tqdm import tqdm
from datetime import datetime
from configs.args import FLAGS
from torch.utils.data import DataLoader
from pathlib import Path
from models.cnn import UNet #
from dataloaders.cnn_dataloader .provider import get_dataset
from torch_utils.visualization import *

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

    model = UNet(kwargs=model_params_config).to(device)
    model.load_state_dict(torch.load(flags.pretrained_dict_path, map_location=device))

    dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Validation after each epoch
    val_running_loss = 0.0
    val_running_mpjpe = 0.0
    val_tot_batch_size = 0.0

    model.eval()
    with torch.no_grad():
        for val_batch_id, val_sample_batched in tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Total batches: '):
            
            val_inputs = val_sample_batched['event_representation'].to(device)
            val_labels = val_sample_batched['label'].to(device)
            val_gt_landmarks = val_sample_batched['landmarks'].to(device)

            val_output = model(val_inputs.float())

            val_batch_size = val_gt_landmarks.shape[0]
            val_tot_batch_size += val_batch_size

            val_loss = my_mse_loss(val_output, val_labels.float(), device=device, sigma=2)
            val_running_loss += val_loss*val_batch_size

            val_pred_landmarks = get_pred_landmarks(val_output)
            val_temp_mpjpe = calc_mpjpe(val_pred_landmarks, val_gt_landmarks)
            val_running_mpjpe += val_temp_mpjpe*val_batch_size

    with open(flags.txt_path, 'w') as f: 
        f.write('%s:%s\n' % ('val_running_mpjpe', val_running_mpjpe.item()/val_tot_batch_size))
        f.write('%s:%s\n' % ('val_running_loss', val_running_loss.item()/val_tot_batch_size))
