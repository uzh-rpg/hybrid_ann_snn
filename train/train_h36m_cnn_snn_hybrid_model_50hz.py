import pdb 
import torch
import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloaders.h36m_snn_dataloader.provider import get_dataset

from configs.args import *
from torch_utils.utils import *
from torch_utils.network_initialization import *
from torch_utils.visualization import *
from models.hybrid_cnn_snn import Hybrid_CNN_SNN
from spikingjelly.clock_driven import functional

if __name__ == '__main__':
    flags, configfile = FLAGS()

    # get config
    model_init_config = configfile['model_initialization']
    model_params_config = configfile['model_parameters']
    dataloader_config = configfile['dataloader_parameters']
    training_config = configfile['training_parameters']

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
    dhp19_dataset = get_dataset(dataloader_config) 
    train_set = dhp19_dataset.get_train_dataset() 
    val_set = dhp19_dataset.get_test_dataset() 

    batch_size = training_config.getint('batch_size')
    params = {'batch_size': batch_size,
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
    model = Hybrid_CNN_SNN(kwargs=model_params_config).to(device)

    if flags.init_with_pretrained:
        print('Initialize CNN and SNN with pretrained weights, CNN weights frozen.')
        model.apply(init_weights_div_last_layer) # initialize all weights with Xavier initialization with last layer weights divided.

        h36m_hybrid_rgb_path = '/home/asude/thesis_final/master_thesis/experiments/checkpoints/HYBRID/2023-02-27_17-57-56_HYBRID_CNN_SNN_H36M_50Hz_lr0.0001_BS8_tau3.0_thr1.0_outputdecay0.8_200ms_epoch6.pt'
        hybrid_rgb_state_dict = torch.load(h36m_hybrid_rgb_path, map_location=device)
        final_dict = {}
        for key_name in hybrid_rgb_state_dict.keys():
            if 'conv1_cnn' not in key_name:
                final_dict[key_name] = hybrid_rgb_state_dict[key_name]
        
        model.load_state_dict(final_dict, strict=False)

        cnn_param_path = model_init_config['cnn_pretrained_dict_path']
        # snn_param_path = model_init_config['snn_pretrained_dict_path']
        
        cnn_final_dict = rename_cnn_model_params(cnn_param_path, device) # Rename parameters of pure models to match hybrid layer names.
        # snn_final_dict = rename_snn_model_params(snn_param_path, device) # Rename parameters of pure models to match hybrid layer names.
        
        model.load_state_dict(cnn_final_dict, strict=False) # Overwrite CNN weights.
        # model.load_state_dict(snn_final_dict, strict=False) # Overwrite SNN weights.
        
        freeze_cnn_weights(model) # Freeze CNN weights for hybrid training.
    else: 
        print('ONLY CNN is initialized with presaved weights from config file and frozen, SNN weights are initialized randomly.')
        model.apply(init_weights_div_last_layer) # initialize all weights with Xavier initialization with last layer weights divided.
        
        cnn_param_path = model_init_config['cnn_pretrained_dict_path']
        
        cnn_final_dict = rename_cnn_model_params(cnn_param_path, device) # Rename parameters of pure models to match hybrid layer names.
        
        model.load_state_dict(cnn_final_dict, strict=False) # Overwrite CNN weights.
        
        freeze_cnn_weights(model) # Freeze CNN weights for hybrid training.


    lr_value = training_config.getfloat('lr')
    optimizer = Adam(model.parameters(), lr=lr_value)

    if save_runs: 
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        comment = '_HYBRID_CNN_SNN_H36M_50Hz_lr{}_BS{}_tau{}_thr{}_outputdecay{}_'.format(lr_value, batch_size, model_params_config.getfloat('tau'), model_params_config.getfloat('v_threshold'), model_params_config.getfloat('output_decay')) + flags.tb_name
        tb = SummaryWriter(comment=comment)
        print('Saving experiments on tensorboard with name, ', comment)
        
    step_counter = 0
    val_step_counter = 0

    num_bins = training_config.getint('num_bins')
    num_labels = training_config.getint('num_labels')
    # assert num_labels == 5, 'Only 20ms labels are supported at the moment.'
    for epoch in range(training_config.getint('num_epochs')):
        running_loss = 0.0
        running_mpjpe = 0.0
        running_mpjpe_avg = 0.0
        mpjpe_list_seq = np.zeros(num_labels)

        model.train()
        print('train mode activated')

        tot_batch_size = 0
        for batch_id, sample_batched in enumerate(train_loader):

            inputs_cnn = sample_batched['image'].to(device)
            inputs = sample_batched['event_representation'].to(device)
            labels = sample_batched['label'].to(device)
            gt_landmarks = sample_batched['landmarks'].to(device)

            sample_batch_size = inputs.shape[0]
            tot_batch_size += sample_batch_size
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # reset neuron states
            functional.reset_net(model)  

            loss = 0
            mpjpe_avg = 0
            for i in range(num_bins):
                if i == 0:
                    output, cnn_output = model(inputs[:,i].float(), inputs_cnn.float(), v_init=True) 
                else: 
                    output = model(inputs[:,i].float(), v_init=False)
                # if i%2:
                # j = int(i/2)
                j = i
                loss += my_mse_loss(output, labels[:, j].float(), sigma=2, device=device)/num_labels
                pred_landmarks = get_pred_landmarks(output)
                mpjpe_temp = calc_mpjpe(pred_landmarks, gt_landmarks[:, j].float()).item()
                mpjpe_avg += mpjpe_temp/num_labels
                mpjpe_list_seq[j] += mpjpe_temp*sample_batch_size
                        
            # last sample
            mpjpe = mpjpe_temp

            # backward + optimize + scheduler
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*sample_batch_size
            running_mpjpe_avg += mpjpe_avg*sample_batch_size
            running_mpjpe += mpjpe*sample_batch_size

            # plot and save params every 1000th iteration when saving is true
            if (not ((batch_id + 1) % 1000)) and save_runs:
                print('batch {}  loss {}  mpjpe {}'.format(batch_id, loss.item(), mpjpe))

                # save gradient flow plot to tensorboard
                image = plot_grad_flow(model.named_parameters(), lr_value)
                tb.add_image('Gradient flow every 1000 samples', image, step_counter)
                plt.close()

                # save first batch predictions plot to tensorboard
                # plot the last prediction from the network and the 10th label overlaid on the entire voxel time of 100ms
                image = plot_batch_label_pred_tb_h36m(sample_batched, output.to('cpu'), mode='hybrid', grid_shape=(2,4), cnn_preds=cnn_output)
                tb.add_image('Training batch predictions every 1000 samples', image, step_counter)
                plt.close()

                # plot mpjpe curve over time 
                fig, ax = plt.subplots()
                ax.plot(mpjpe_list_seq/tot_batch_size)
                ax.set_xlabel('Bin number')
                ax.set_ylabel('MPJPE')
                img = plot_to_tensor()
                tb.add_image('MPJPE curve of sample', img, step_counter)
                plt.close()

                # Save average training loss
                tb.add_scalar("Avg training MSE loss every 1000 samples", running_loss/tot_batch_size, step_counter)

                # Save average mpjpe metric
                tb.add_scalar("Avg training MPJPE every 1000 samples", running_mpjpe_avg/tot_batch_size, step_counter)

                # Save average training loss
                tb.add_scalar("Last bin training MPJPE every 1000 samples", running_mpjpe/tot_batch_size, step_counter)

                step_counter += 1
                    
                running_loss = 0.0
                running_mpjpe = 0.0
                running_mpjpe_avg = 0.0
                mpjpe_list_seq = np.zeros(num_labels)

                tot_batch_size = 0.0
            
            # GET INTERMEDIATE VALIDATION SCORES 
            if (not (batch_id + 1)%(20*1000)) and save_runs:
                if save_params:
                    model_param_path = flags.output_path + dt_string + comment + '_epoch{}_iter{}'.format(epoch, batch_id) + '.pt'
                    torch.save(model.state_dict(), model_param_path)

                model.eval()
                print('eval mode activated')
                with torch.no_grad(): # computational graphs are not created

                    val_running_loss = 0.0
                    val_running_mpjpe = 0.0
                    val_running_mpjpe_avg = 0.0
                    val_mpjpe_list_seq = np.zeros(num_labels)

                    val_tot_batch_size = 0                        

                    for val_batch_id, val_sample_batched in enumerate(val_loader):
                        
                        val_inputs_cnn = val_sample_batched['image'].to(device)
                        val_inputs = val_sample_batched['event_representation'].to(device)
                        val_labels = val_sample_batched['label'].to(device)
                        val_gt_landmarks = val_sample_batched['landmarks'].to(device)

                        val_sample_batch_size = val_inputs.shape[0]
                        val_tot_batch_size += val_sample_batch_size

                        # reset neurons
                        functional.reset_net(model)

                        # forward + loss
                        val_loss = 0
                        val_mpjpe_avg = 0
                        for i in range(num_bins):
                            if i == 0: 
                                val_output, val_cnn_output = model(val_inputs[:, i].float(), val_inputs_cnn.float(), v_init=True) # todo
                            else:                   
                                val_output = model(val_inputs[:, i].float(), v_init=False)
                            # if i%2: 
                            # j = int(i/2)
                            j = i
                            val_loss += my_mse_loss(val_output, val_labels[:, j].float(), sigma=2, device=device)/num_labels
                            val_pred_landmarks = get_pred_landmarks(val_output)
                            val_mpjpe_temp = calc_mpjpe(val_pred_landmarks, val_gt_landmarks[:, j].float()).item()
                            val_mpjpe_avg += val_mpjpe_temp/num_labels
                            val_mpjpe_list_seq[j] += val_mpjpe_temp*val_sample_batch_size

                        # last mpjpe
                        val_mpjpe = val_mpjpe_temp

                        val_running_mpjpe_avg += val_mpjpe_avg*val_sample_batch_size
                        val_running_mpjpe += val_mpjpe*val_sample_batch_size
                        val_running_loss += val_loss.item()*val_sample_batch_size

                        # look at 2000 iterations and quit
                        if val_batch_id == 1000 and save_runs:
                            image = plot_batch_label_pred_tb_h36m(val_sample_batched, val_output.to('cpu'), mode='hybrid', grid_shape=(2,4), cnn_preds=val_cnn_output)
                            tb.add_image('Validation batch predictions every 1000 samples', image, val_step_counter)
                            plt.close()
                        # if val_batch_id == 2000 and save_runs:
                    # save first validation batch predictions plot to tensorboard

                    print('batch {}  loss {}  mpjpe {}'.format(val_batch_id, val_loss.item(), val_mpjpe))

                    #  Save average validation mse loss
                    tb.add_scalar("Avg validation MSE loss", val_running_loss / val_tot_batch_size, val_step_counter)

                    #  Save average validation mse loss
                    tb.add_scalar("Avg validation MPJPE score", val_running_mpjpe_avg / val_tot_batch_size, val_step_counter)

                    # Save average validation mpjpe metric
                    tb.add_scalar("Avg last bin validation MPJPE scores", val_running_mpjpe / val_tot_batch_size, val_step_counter)
            
                    fig, ax = plt.subplots()
                    ax.plot(val_mpjpe_list_seq/val_tot_batch_size)
                    ax.set_xlabel('Bin number')
                    ax.set_ylabel('MPJPE')
                    img = plot_to_tensor()
                    tb.add_image('Validation MPJPE curve of sample', img, val_step_counter)
                    plt.close()

                    val_step_counter += 1

                model.train()
                print('Train mode activated.')

        # save model at the end of each epoch if param saving is true
        if save_params:
            model_param_path = flags.output_path + dt_string + comment + '_epoch{}'.format(epoch) + '.pt'
            torch.save(model.state_dict(), model_param_path)
                
        model.eval()
        print('eval mode activated')

        # evaluate model at every 1000th iteration
        with torch.no_grad(): # computational graphs are not created

            val_running_loss = 0.0
            val_running_mpjpe = 0.0
            val_running_mpjpe_avg = 0.0
            val_mpjpe_list_seq = np.zeros(num_labels)

            val_tot_batch_size = 0                        

            for val_batch_id, val_sample_batched in enumerate(val_loader):
                
                val_inputs_cnn = val_sample_batched['image'].to(device)
                val_inputs = val_sample_batched['event_representation'].to(device)
                val_labels = val_sample_batched['label'].to(device)
                val_gt_landmarks = val_sample_batched['landmarks'].to(device)

                val_sample_batch_size = val_inputs.shape[0]
                val_tot_batch_size += val_sample_batch_size

                # reset neurons
                functional.reset_net(model)

                # forward + loss
                val_loss = 0
                val_mpjpe_avg = 0
                for i in range(num_bins):
                    if i == 0: 
                        val_output, val_cnn_output = model(val_inputs[:, i].float(), val_inputs_cnn.float(), v_init=True) # todo
                    else:                   
                        val_output = model(val_inputs[:, i].float(), v_init=False)
                    # if i%2: 
                    # j = int(i/2)
                    j = i
                    val_loss += my_mse_loss(val_output, val_labels[:, j].float(), sigma=2, device=device)/num_labels
                    val_pred_landmarks = get_pred_landmarks(val_output)
                    val_mpjpe_temp = calc_mpjpe(val_pred_landmarks, val_gt_landmarks[:, j].float()).item()
                    val_mpjpe_avg += val_mpjpe_temp/num_labels
                    val_mpjpe_list_seq[j] += val_mpjpe_temp*val_sample_batch_size

                # last mpjpe
                val_mpjpe = val_mpjpe_temp

                val_running_mpjpe_avg += val_mpjpe_avg*val_sample_batch_size
                val_running_mpjpe += val_mpjpe*val_sample_batch_size
                val_running_loss += val_loss.item()*val_sample_batch_size

                # plot a random sample of validation predictions 
                if val_batch_id == 1000 and save_runs:
                    # save first validation batch predictions plot to tensorboard
                    image = plot_batch_label_pred_tb_h36m(val_sample_batched, val_output.to('cpu'), mode='hybrid', grid_shape=(2,4), cnn_preds=val_cnn_output)
                    tb.add_image('Validation batch predictions every 1000 samples', image, val_step_counter)
                    plt.close()

            # save validation parameters on tensorboard 
            if save_runs:
                print('batch {}  loss {}  mpjpe {}'.format(val_batch_id, val_loss.item(), val_mpjpe))

                #  Save average validation mse loss
                tb.add_scalar("Avg validation MSE loss", val_running_loss / val_tot_batch_size, val_step_counter)

                #  Save average validation mse loss
                tb.add_scalar("Avg validation MPJPE score", val_running_mpjpe_avg / val_tot_batch_size, val_step_counter)

                # Save average validation mpjpe metric
                tb.add_scalar("Avg last bin validation MPJPE scores", val_running_mpjpe / val_tot_batch_size, val_step_counter)
                
                fig, ax = plt.subplots()
                ax.plot(val_mpjpe_list_seq/val_tot_batch_size)
                ax.set_xlabel('Bin number')
                ax.set_ylabel('MPJPE')
                img = plot_to_tensor()
                tb.add_image('Validation MPJPE curve of sample', img, val_step_counter)
                plt.close()

                val_step_counter += 1

        print('Finished epoch.')
    print('Finished training!')


    if save_runs:
        tb.flush()