import pdb 
import torch
import numpy as np
import torch.nn as nn

from datetime import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloaders.snn_dataloader.provider import get_dataset

from configs.args import *
from torch_utils.utils import *
from torch_utils.network_initialization import *
from torch_utils.visualization import *
from models.hybrid_cnn_rnn import Hybrid_CNN_RNN
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

    model = Hybrid_CNN_RNN(model_params_config).to(device)

    if flags.init_with_pretrained:
        print('Initialize CNN and RNN with pretrained weights, CNN weights frozen.')
        model.apply(init_weights_div_last_layer) # initialize all weights with Xavier initialization with last layer weights divided.
        
        cnn_param_path = model_init_config['cnn_pretrained_dict_path']
        rnn_param_path = model_init_config['rnn_pretrained_dict_path']
    
        assert Path(rnn_param_path).exists(), 'RNN pretrained weight path does not exist, please set it in the config file.'

        cnn_final_dict = rename_cnn_model_params(cnn_param_path, device) # Rename parameters of pure models to match hybrid layer names.
        
        model.load_state_dict(cnn_final_dict, strict=False) # Overwrite CNN weights.
        model.load_state_dict(torch.load(rnn_param_path), strict=False) # Overwrite SNN weights.
        
        freeze_cnn_weights(model) # Freeze CNN weights for hybrid training.
    
    else: 
        print('ONLY CNN is initialized with presaved weights from config file and frozen, RNN weights are initialized randomly.')
        model.apply(init_weights_div_last_layer) # initialize all weights with Xavier initialization with last layer weights divided.
        
        cnn_param_path = model_init_config['cnn_pretrained_dict_path']
        
        cnn_final_dict = rename_cnn_model_params(cnn_param_path, device) # Rename parameters of pure models to match hybrid layer names.
        
        model.load_state_dict(cnn_final_dict, strict=False) # Overwrite CNN weights.
        
        freeze_cnn_weights(model) # Freeze CNN weights for hybrid training.

    lr_value = training_config.getfloat('lr')
    optimizer = Adam(model.parameters(), lr=lr_value)

    if save_runs: 
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        comment = '_HYBRID_CNN_RNN_' + flags.tb_name
        tb = SummaryWriter(comment=comment)

    step_counter = 0

    num_bins = training_config.getint('num_bins')
    for epoch in range(training_config.getint('num_epochs')):
        running_loss = 0.0
        running_mpjpe = 0.0
        running_mpjpe_avg10 = 0.0
            
        model.train()
        print('train mode activated')

        tot_batch_size = 0
        for batch_id, sample_batched in enumerate(train_loader):
            print(batch_id)
            
            inputs = sample_batched['event_representation'].to(device)
            inputs_cnn = sample_batched['cnn_input'].to(device)
            labels = sample_batched['label'].to(device)
            gt_landmarks = sample_batched['landmarks'].to(device)

            sample_batch_size = inputs.shape[0]
            tot_batch_size += sample_batch_size
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # reset neuron states
            functional.reset_net(model)  

            loss = 0
            mpjpe_avg10 = 0
            mpjpe_list_sample = []
            prev_states = None
            for i in range(num_bins):
                if i == 0: 
                    output, states, cnn_output = model(inputs[:,i].float(), inputs_cnn.float(), v_init=True, prev_states=prev_states)
                else:
                    output, states = model(inputs[:,i].float(), v_init=False, prev_states=prev_states)

                loss += my_mse_loss(output, labels[:, i].float(), sigma=2, device=device)/num_bins
                pred_landmarks = get_pred_landmarks(output)
                mpjpe_temp = calc_mpjpe(pred_landmarks, gt_landmarks[:, :, i]).item()
                mpjpe_avg10 += mpjpe_temp/num_bins

                # register hooks at every 1000th iteration for the last num_bin when saving runs is true 
                if (not ((batch_id + 1) % 1000)) and save_runs:
                    mpjpe_list_sample.append(mpjpe_temp)
            
                prev_states = states           # last sample
            
            mpjpe = mpjpe_temp

            # backward + optimize + scheduler
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*sample_batch_size
            running_mpjpe_avg10 += mpjpe_avg10*sample_batch_size
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
                image = plot_batch_label_pred_tb(sample_batched, output.to('cpu'), points='all', grid_shape=(2,4), plot_init=True)
                tb.add_image('Training batch predictions every 1000 samples', image, step_counter)
                plt.close()

                # plot mpjpe curve over time 
                fig, ax = plt.subplots()
                ax.plot(mpjpe_list_sample)
                ax.set_xlabel('Bin number')
                ax.set_ylabel('MPJPE')
                img = plot_to_tensor()
                tb.add_image('MPJPE curve of sample', img, step_counter)
                plt.close()

                # Save average training loss
                tb.add_scalar("Avg training MSE loss every 1000 samples", running_loss/tot_batch_size, step_counter)

                # Save average mpjpe metric
                tb.add_scalar("Avg training MPJPE every 1000 samples", running_mpjpe_avg10/tot_batch_size, step_counter)

                # Save average training loss
                tb.add_scalar("Last bin training MPJPE every 1000 samples", running_mpjpe/tot_batch_size, step_counter)

                step_counter += 1
                    
                running_loss = 0.0
                running_mpjpe = 0.0
                running_mpjpe_avg10 = 0.0

                tot_batch_size = 0.0

        # save model at the end of each epoch if param saving is true
        if save_params:
            model_param_path = flags.output_path + dt_string + comment + '{}'.format(epoch) + '.pt'
            torch.save(model.state_dict(), model_param_path)
                
        model.eval()
        print('eval mode activated')

        # evaluate model at every 1000th iteration
        with torch.no_grad(): # computational graphs are not created

            val_running_loss = 0.0
            val_running_mpjpe = 0.0
            val_running_mpjpe_avg10 = 0.0
            # val_mpjpe_list = np.zeros((10))

            val_tot_batch_size = 0                            

            for val_batch_id, val_sample_batched in enumerate(val_loader):
                print(val_batch_id)
                
                val_inputs = val_sample_batched['event_representation'].to(device)
                val_inputs_cnn = val_sample_batched['cnn_input'].to(device)
                val_labels = val_sample_batched['label'].to(device)
                val_gt_landmarks = val_sample_batched['landmarks'].to(device)

                val_sample_batch_size = val_inputs.shape[0]
                val_tot_batch_size += val_sample_batch_size

                # reset neurons
                functional.reset_net(model)

                # forward + loss
                val_loss = 0
                val_mpjpe_avg10 = 0
                val_mpjpe_list = []

                val_prev_states = None
                for i in range(num_bins):
                    if i == 0: 
                        val_output, val_states, val_cnn_output = model(val_inputs[:, i].float(), val_inputs_cnn.float(), v_init=True) 
                    else:
                        val_output, val_states = model(val_inputs[:, i].float(), v_init=False)

                    val_loss += my_mse_loss(val_output, val_labels[:, i].float(), sigma=2, device=device)/num_bins
                    val_pred_landmarks = get_pred_landmarks(val_output)
                    val_mpjpe_temp = calc_mpjpe(val_pred_landmarks, val_gt_landmarks[:, :, i]).item()
                    val_mpjpe_avg10 += val_mpjpe_temp/num_bins

                    val_prev_states = val_states

                    if val_batch_id == 1000:
                        val_mpjpe_list.append(val_mpjpe_temp.item())

                # last mpjpe
                val_mpjpe = val_mpjpe_temp

                val_running_mpjpe_avg10 += val_mpjpe_avg10*val_sample_batch_size
                val_running_mpjpe += val_mpjpe*val_sample_batch_size
                val_running_loss += val_loss.item()*val_sample_batch_size

                # plot a random sample of validation predictions 
                if val_batch_id == 1000:
                    # save first validation batch predictions plot to tensorboard
                    image = plot_batch_label_pred_tb(val_sample_batched, val_output.to('cpu'), points='all', grid_shape=(2,4), plot_init=True)
                    tb.add_image('Validation batch predictions every 1000 samples', image, epoch)
                    plt.close()

            # save validation parameters on tensorboard 
            if save_runs:
                print('batch {}  loss {}  mpjpe {}'.format(val_batch_id, val_loss.item(), val_mpjpe))

                #  Save average validation mse loss
                tb.add_scalar("Avg validation MSE loss", val_running_loss / val_tot_batch_size, epoch)

                #  Save average validation mse loss
                tb.add_scalar("Avg validation MPJPE score", val_running_mpjpe_avg10 / val_tot_batch_size, epoch)

                # Save average validation mpjpe metric
                tb.add_scalar("Avg last bin validation MPJPE scores", val_running_mpjpe / val_tot_batch_size, epoch)
        
        print('Finished epoch.')
    print('Finished training!')

    if save_runs:
        tb.flush()