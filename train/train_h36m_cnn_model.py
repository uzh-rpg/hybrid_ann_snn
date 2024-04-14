'''
Train the CNN network with SNN compatible data. 
'''
import torch.nn as nn
from configs.args import *

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam

from datetime import datetime
from torch_utils.network_initialization import init_weights, init_weights_div_last_layer
from models.cnn import UNet
from dataloaders.h36m_cnn_dataloader.provider import get_dataset
from torch_utils.visualization import *

if __name__ == '__main__':
    flags, configfile = FLAGS()

    # get config
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
    print('Running on', device)

    # PREPARE DATA LOADER
    dhp19_dataset = get_dataset(dataloader_config) 
    train_set = dhp19_dataset.get_train_dataset() 
    val_set = dhp19_dataset.get_test_dataset() 

    img_type = dataloader_config['img_type']

    # print(train_set.__len__())
    # print(val_set.__len__())
    # pdb.set_trace()

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

    model = UNet(kwargs=model_params_config).to(device)

    if flags.init_with_pretrained:
        print('Initialized with presaved weights from config file.')
        assert Path(model_init_config['cnn_pretrained_dict_path'].exists()), 'Pretrained weight path does not exists, please spesificy it in the config file.'
        model.load_state_dict(torch.load(model_init_config['cnn_pretrained_dict_path']), strict=True) # Overwrite CNN weights.
    else: 
        print('Initialize with random normal distribution of weights.')
        # model.apply(init_weights)
        model.apply(init_weights_div_last_layer)

    lr_value = training_config.getfloat('lr')
    optimizer = Adam(model.parameters(), lr=lr_value)

    if save_runs: 
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        comment = '_CNN_H36M_lr{}_BS{}_'.format(lr_value, batch_size) + flags.tb_name
        tb = SummaryWriter(comment=comment)
        print('Saving results on tensorboard with name ', comment)

    step_counter = 0
    for epoch in range(training_config.getint('num_epochs')):
        running_loss = 0.0
        running_mpjpe = 0.0
        tot_batch_size = 0.0

        model.train()
        for batch_id, sample_batched in enumerate(train_loader):
            inputs = sample_batched['image'].to(device)
            labels = sample_batched['label'].to(device)
            gt_landmarks = sample_batched['landmarks'].to(device)

            sample_batch_size = gt_landmarks.shape[0]
            tot_batch_size += sample_batch_size

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss
            output = model(inputs.float()) # inputs already float
            loss = my_mse_loss(output, labels.float(), device=device, sigma=2)

            # backward + optimize + scheduler
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*sample_batch_size

            pred_landmarks = get_pred_landmarks(output)
            temp_mpjpe = calc_mpjpe(pred_landmarks, gt_landmarks.float())
            running_mpjpe += temp_mpjpe*sample_batch_size

            if (not ((batch_id + 1) % 1000)) and save_runs:
                print('batch {}  loss {}'.format(batch_id, loss.item()))

                # save gradient flow plot to tensorboard
                image = plot_grad_flow(model.named_parameters(), lr_value)
                tb.add_image('gradient flow at start of each epoch', image, step_counter)
                plt.close()

                # save first batch predictions plot to tensorboard
                image = plot_batch_label_pred_tb_h36m(sample_batched, output.to('cpu'), grid_shape=(2,4), mode='cnn', img_color=img_type)
                tb.add_image('batch predictions at start of each epoch', image, step_counter)
                plt.close()

                # Save average training loss
                tb.add_scalar("avg training loss per epoch", running_loss/tot_batch_size, step_counter)

                # Save average mpjpe metric
                tb.add_scalar("avg mpjpe accuracy per epoch", running_mpjpe/tot_batch_size, step_counter)
        
                step_counter += 1
                running_loss = 0.0
                running_mpjpe = 0.0
                tot_batch_size = 0.0

        print('epoch {}'.format(epoch))

        if save_params:
            # save model at the end of each epoch
            model_param_path = flags.output_path + dt_string + comment + '{}'.format(epoch) + '.pt'
            torch.save(model.state_dict(), model_param_path)

        # Validation after each epoch
        val_running_loss = 0.0
        val_running_mpjpe = 0.0
        val_tot_batch_size = 0.0

        val_running_mpjpe_corrupt = 0.0
        val_tot_batch_size_corrupt = 0.0

        model.eval()

        with torch.no_grad():
            for val_batch_id, val_sample_batched in enumerate(val_loader):
                val_inputs = val_sample_batched['image'].to(device)
                val_labels = val_sample_batched['label'].to(device)
                val_gt_landmarks = val_sample_batched['landmarks'].to(device)
                val_action_name = val_sample_batched['action_name']
                val_subject_name = val_sample_batched['subject_name']

                val_output = model(val_inputs.float())
                val_loss = my_mse_loss(val_output, val_labels.float(), device=device, sigma=2)

                val_batch_size = val_gt_landmarks.shape[0]
                val_tot_batch_size += val_batch_size

                val_running_loss += val_loss.item()*val_batch_size

                val_pred_landmarks = get_pred_landmarks(val_output)
                temp_mpjpe = calc_mpjpe(val_pred_landmarks, val_gt_landmarks.float())
                val_running_mpjpe += temp_mpjpe*val_batch_size

                if val_subject_name == 'S9':
                    if val_action_name == 'Greeting' or val_action_name =='SittingDown 1' or val_action_name=='Waiting 1':
                        val_running_mpjpe_corrupt += temp_mpjpe*val_batch_size
                        val_tot_batch_size_corrupt += val_batch_size

                if val_batch_id == 0 and save_runs:
                    print('val batch {}'.format(val_batch_id))

                    # save first validation batch predictions plot to tensorboard
                    image = plot_batch_label_pred_tb_h36m(val_sample_batched, val_output.to('cpu'), grid_shape=(2,4), mode='cnn', img_color=img_type)
                    tb.add_image('validation batch predictions at start of each epoch', image, epoch)

        if save_runs:

            #  Save average validation mse loss
            tb.add_scalar("avg validation loss per epoch", val_running_loss/val_tot_batch_size, epoch)

            # Save average validation mpjpe metric
            tb.add_scalar("avg validation mpjpe accuracy per epoch", val_running_mpjpe/val_tot_batch_size, epoch)
            
            # Save average validation mpjpe metric
            tb.add_scalar("avg validation mpjpe accuracy per epoch without corrupted files", (val_running_mpjpe - val_running_mpjpe_corrupt)/(val_tot_batch_size - val_tot_batch_size_corrupt), epoch)
              
        print('Finished training for single lr')
    print('Finished training!')

    if save_runs:
        tb.flush()