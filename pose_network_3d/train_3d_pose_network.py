import pdb
import torch
import numpy as np
import torch.nn as nn
import h5py
import os 

from datetime import datetime
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.args import *
from torch_utils.utils import *
from torch_utils.ranger import Ranger
from torch_utils.utils_3d import calc_3D_mpjpe
from torch_utils.network_initialization import *
from torch_utils.visualization import *
from pose_network_3d.models.pose_network_3D import TemporalModelOptimized1f
from spikingjelly.clock_driven import functional
from pose_network_3d.dataloaders.pose_network_dataloader.provider import get_dataset

if __name__ == '__main__':
    flags, configfile = FLAGS()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
    print(torch.cuda.is_available())

    # get config
    dataloader_config = configfile['dataloader_parameters']
    training_config = configfile['training_parameters']
    inference_config = configfile['inference_parameters']
    model_params_config = configfile['model_parameters']

    save_runs = flags.save_runs
    save_params = flags.save_params

    _seed_ = 2022
    np.random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True    
    torch.backends.cudnn.allow_tf32 = True

    # SET DEVICE
    device = torch.device(flags.device)

    # PREPARE DATA LOADER
    dhp19_dataset = get_dataset(dataloader_config['dataset_path'], dataloader_config.getint('receptive_field'))
    train_set = dhp19_dataset.get_train_dataset() 
    val_set = dhp19_dataset.get_test_dataset() 
    print(val_set.__len__())

    params = {'batch_size': training_config.getint('batch_size'),
            'num_workers': training_config.getint('num_workers'),
            'pin_memory': True}

    train_loader = DataLoader(
        train_set,
        shuffle = True,
        **params
    )

    val_loader = DataLoader(
        val_set,
        shuffle = False,
        **params, 
    )

    num_joints_in = model_params_config.getint('num_joints_in')
    in_features = model_params_config.getint('in_features')
    num_joints_out = model_params_config.getint('num_joints_out')
    filter_widths = model_params_config['filter_widths']
    filter_widths = [int(x) for x in filter_widths[1:-1].split(',')]

    model = TemporalModelOptimized1f(num_joints_in=num_joints_in, in_features=in_features, num_joints_out=num_joints_out, filter_widths=filter_widths).to(device)

    model.apply(init_weights)
    
    lr_value = training_config.getfloat('lr')
    batch_size = training_config.getint('batch_size')
    epochs = training_config.getint('num_epochs')
    # lr_decay = 0.95
    initial_momentum = 0.1
    final_momentum = 0.001
    # TODO
    optimizer = Ranger(model.parameters(), lr=lr_value)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=epochs) # LR COSINE 
    # optimizer = Adam(model.parameters(), lr=lr_value)

    if save_runs: 
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        comment = '_3D_pose_network_' + 'lr{}-'.format(lr_value) + 'BS{}-'.format(batch_size) + flags.tb_name 
        tb = SummaryWriter(comment=comment)
        print('saving to tb with name ', comment)

    for epoch in range(epochs):
        running_mpjpe = 0.0
            
        model.train()
        print('train mode activated')

        tot_batch_size = 0
        for batch_id, sample_batched in enumerate(train_loader):

            inputs = sample_batched['estimates_2d'].to(device)
            label = sample_batched['gt_3d'].to(device)
            gt_3d_mask = sample_batched['gt_3d_nan_mask'].to(device)

            # assert inputs.isfinite().all()

            sample_batch_size = inputs.shape[0]
            tot_batch_size += sample_batch_size
            
            # zero the parameter gradients
            optimizer.zero_grad()

            pred_3d = model(inputs.float())
            loss = calc_3D_mpjpe(pred_3d, label, gt_3d_mask)

            print(batch_id)
            
            mpjpe = loss.detach()
            running_mpjpe += mpjpe.item()*sample_batch_size

            # backward + optimize + scheduler
            loss.backward()
            optimizer.step()

        if save_runs: 
            print('batch {}, mpjpe {}'.format(batch_id, mpjpe.item()))
            tb.add_scalar('Train loss (3D MPJPE)', running_mpjpe/tot_batch_size, epoch)
            
            image = plot_grad_flow(model.named_parameters())
            tb.add_image('Gradient flow', image, epoch)
            plt.close()

        # save model at the end of each epoch if param saving is true
        if save_params:
            model_param_path = flags.output_path + dt_string + comment + '{}'.format(epoch) + '.pt'
            torch.save(model.state_dict(), model_param_path)
                
        model.eval()
        print('eval mode activated')

        # evaluate model at every 1000th iteration
        with torch.no_grad(): # computational graphs are not created

            val_running_mpjpe = 0.0

            val_tot_batch_size = 0                            

            for val_batch_id, val_sample_batched in enumerate(val_loader):
                print(val_batch_id)
                
                val_inputs = val_sample_batched['estimates_2d'].to(device)
                val_label = val_sample_batched['gt_3d'].to(device)
                val_mask = val_sample_batched['gt_3d_nan_mask'].to(device)

                val_sample_batch_size = val_inputs.shape[0]
                val_tot_batch_size += val_sample_batch_size

                val_pred_3d = model(val_inputs.float())
                val_mpjpe = calc_3D_mpjpe(val_pred_3d, val_label, val_mask)

                val_running_mpjpe += val_mpjpe*val_sample_batch_size

            if save_runs:
                print('batch {}, mpjpe {}'.format(val_batch_id, val_mpjpe))
                tb.add_scalar('Val loss (3D MPJPE)', val_running_mpjpe/val_tot_batch_size, epoch)
        
        # cosin annealing
        scheduler.step()
        lr = scheduler.get_lr()[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = initial_momentum * np.exp(-(epoch+1) / epochs * np.log(initial_momentum / final_momentum))
        model.set_bn_momentum(momentum)
        model.set_KA_bn(momentum)
        model.set_expand_bn(momentum)
        model.set_dilation_bn(momentum)

        print('Finished epoch.')
    print('Finished training!')


    if save_runs:
        tb.flush()

