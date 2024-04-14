import configparser
# TODO update 
### SET PATHS ###
dataset_path = '/media/mathias/moar/asude/dhp19_h5_event_files/all_events/346x260/'
P_mat_path = '/home/rslsync/Resilio Sync/DHP19/P_matrices/'
save_root_dir = '/home/asude/thesis_final/master_thesis/configs/'
checkpoint_dir = '/media/mathias/moar/asude/checkpoints/'

######## HYBRID CNN-SNN ########
config = configparser.ConfigParser()

config.add_section('model_initialization')
config.set('model_initialization', 'cnn_pretrained_dict_path', checkpoint_dir + 'CNN_CONST_COUNT_7500_10x2=20channels_camview3_5.pt')
config.set('model_initialization', 'snn_pretrained_dict_path', checkpoint_dir + 'SNN_10x10=100ms_tau2.0_vthr1.0_camview3_7.pt')

config.add_section('model_parameters')
config.set('model_parameters', 'activation_type', 'lif_init')
config.set('model_parameters', 'v_threshold', '1.0')
config.set('model_parameters', 'tau', '3.0')
# config.set('model_parameters', 'v_reset', 'None')
config.set('model_parameters', 'detach_reset', 'False')
config.set('model_parameters', 'init_mp_dicts', 'False')
config.set('model_parameters', 'output_decay', '0.8')

config.add_section('dataloader_parameters')
config.set('dataloader_parameters', 'camera_views', '[3]')
config.set('dataloader_parameters', 'sequence_length', '100')
config.set('dataloader_parameters', 'step_size', '100')
config.set('dataloader_parameters', 'bin_length_per_stack', '10')
config.set('dataloader_parameters', 'cnn_input', 'True')
config.set('dataloader_parameters', 'constant_count', 'True')
config.set('dataloader_parameters', 'constant_duration', 'False')
config.set('dataloader_parameters', 'dataset_path', dataset_path)
config.set('dataloader_parameters', 'P_mat_path', P_mat_path)
config.set('dataloader_parameters', 'downsample', 'False')
config.set('dataloader_parameters', 'ev_representation', 'stacked_hist')

config.add_section('training_parameters')
config.set('training_parameters', 'num_epochs', '200')
config.set('training_parameters', 'lr', '0.00005')
config.set('training_parameters', 'batch_size', '2')
config.set('training_parameters', 'num_workers', '4')
config.set('training_parameters', 'num_bins', '10')

config.add_section('inference_parameters')
config.set('inference_parameters', 'batch_size', '4')
config.set('inference_parameters', 'num_workers', '4')
config.set('inference_parameters', 'num_bins', '10')

with open(save_root_dir + r"hybrid_cnn_snn_config.ini", 'w') as configfile:
    config.write(configfile)

########## SNN ############
config = configparser.ConfigParser()

config.add_section('model_initialization')
config.set('model_initialization', 'snn_pretrained_dict_path', checkpoint_dir + 'SNN_10x10=100ms_tau2.0_vthr1.0_camview3_7.pt')

config.add_section('model_parameters')
config.set('model_parameters', 'activation_type', 'lif')
config.set('model_parameters', 'v_threshold', '1.0')
config.set('model_parameters', 'tau', '2.0')
# config.set('model_parameters', 'v_reset', 'None')
config.set('model_parameters', 'detach_reset', 'False')

config.add_section('dataloader_parameters')
config.set('dataloader_parameters', 'cnn_input', 'False')
config.set('dataloader_parameters', 'camera_views', '[3]')
config.set('dataloader_parameters', 'sequence_length', '100')
config.set('dataloader_parameters', 'step_size', '100')
config.set('dataloader_parameters', 'bin_length_per_stack', '10')
config.set('dataloader_parameters', 'dataset_path', dataset_path)
config.set('dataloader_parameters', 'P_mat_path', P_mat_path)
config.set('dataloader_parameters', 'downsample', 'False')
config.set('dataloader_parameters', 'ev_representation', 'stacked_hist')

config.add_section('training_parameters')
config.set('training_parameters', 'num_epochs', '200')
config.set('training_parameters', 'lr', '0.00005')
config.set('training_parameters', 'batch_size', '2')
config.set('training_parameters', 'num_workers', '4')
config.set('training_parameters', 'num_bins', '10')

config.add_section('inference_parameters')
config.set('inference_parameters', 'batch_size', '8')
config.set('inference_parameters', 'num_workers', '4')
config.set('inference_parameters', 'num_bins', '10')

with open(save_root_dir + r"snn_config.ini", 'w') as configfile:
    config.write(configfile)

config = configparser.ConfigParser()

config.add_section('model_initialization')
config.set('model_initialization', 'cnn_pretrained_dict_path', '')

config.add_section('model_parameters')
config.set('model_parameters', 'in_channels', '20')    

config.add_section('dataloader_parameters')
config.set('dataloader_parameters', 'constant_count', 'True')
config.set('dataloader_parameters', 'constant_duration', 'False')
config.set('dataloader_parameters', 'camera_views', '[3]')
config.set('dataloader_parameters', 'sequence_length', '100')
config.set('dataloader_parameters', 'step_size', '100')
config.set('dataloader_parameters', 'num_bins', '10')
config.set('dataloader_parameters', 'dataset_path', dataset_path)
config.set('dataloader_parameters', 'P_mat_path', P_mat_path)
config.set('dataloader_parameters', 'ev_representation', 'stacked_hist')

config.add_section('training_parameters')
config.set('training_parameters', 'num_epochs', '200')
config.set('training_parameters', 'lr', '0.0001')
config.set('training_parameters', 'batch_size', '8')
config.set('training_parameters', 'num_workers', '4')

config.add_section('inference_parameters')
config.set('inference_parameters', 'batch_size', '10')
config.set('inference_parameters', 'num_workers', '4')

with open(save_root_dir + r"cnn_config.ini", 'w') as configfile:
    config.write(configfile)

######### HYBRID CNN-RNN ##############
config = configparser.ConfigParser()

config.add_section('model_initialization')
config.set('model_initialization', 'cnn_pretrained_dict_path', checkpoint_dir + 'CNN_CONST_DURATION_100ms_10x2=20channels_camview3_6.pt')
config.set('model_initialization', 'rnn_pretrained_dict_path', checkpoint_dir + 'RNN_10x10=100ms_camview3_9.pt')

config.add_section('model_parameters')
config.set('model_parameters', 'in_channels', '2')   
config.set('model_parameters', 'out_channels', '13')  
config.set('model_parameters', 'skip_type', 'sum')    
config.set('model_parameters', 'activation', 'leaky_relu')      
config.set('model_parameters', 'recurrent_block_type', 'convlstm')    
config.set('model_parameters', 'num_encoders', '3')    
config.set('model_parameters', 'base_num_channels', '32')      
config.set('model_parameters', 'num_residual_blocks', '2')    
config.set('model_parameters', 'use_upsample_conv', 'False')    
config.set('model_parameters', 'norm', 'BN')      
config.set('model_parameters', 'running_stats', 'True')  
config.set('model_parameters', 'output_decay', '0.8')  

config.add_section('dataloader_parameters')
config.set('dataloader_parameters', 'cnn_input', 'True')
config.set('dataloader_parameters', 'bin_length_per_stack', '10')
config.set('dataloader_parameters', 'constant_count', 'False')
config.set('dataloader_parameters', 'constant_duration', 'True')
config.set('dataloader_parameters', 'camera_views', '[3]')
config.set('dataloader_parameters', 'sequence_length', '100')
config.set('dataloader_parameters', 'step_size', '100')
config.set('dataloader_parameters', 'num_bins', '10')
config.set('dataloader_parameters', 'dataset_path', dataset_path)
config.set('dataloader_parameters', 'P_mat_path', P_mat_path)
config.set('dataloader_parameters', 'ev_representation', 'stacked_hist')

config.add_section('training_parameters')
config.set('training_parameters', 'num_epochs', '200')
config.set('training_parameters', 'lr', '0.0001')
config.set('training_parameters', 'batch_size', '2')
config.set('training_parameters', 'num_workers', '4')
config.set('training_parameters', 'num_bins', '10')

config.add_section('inference_parameters')
config.set('inference_parameters', 'batch_size', '4')
config.set('inference_parameters', 'num_workers', '4')
config.set('inference_parameters', 'num_bins', '10')

with open(save_root_dir + r"hybrid_cnn_rnn_config.ini", 'w') as configfile:
    config.write(configfile)

config = configparser.ConfigParser()

config.add_section('model_initialization')
config.set('model_initialization', 'rnn_pretrained_dict_path', '')

config.add_section('model_parameters')
config.set('model_parameters', 'in_channels', '2')   
config.set('model_parameters', 'out_channels', '13')  
config.set('model_parameters', 'skip_type', 'sum')    
config.set('model_parameters', 'activation', 'leaky_relu')      
config.set('model_parameters', 'recurrent_block_type', 'convlstm')    
config.set('model_parameters', 'num_encoders', '3')    
config.set('model_parameters', 'base_num_channels', '32')      
config.set('model_parameters', 'num_residual_blocks', '2')    
config.set('model_parameters', 'use_upsample_conv', 'False')    
config.set('model_parameters', 'norm', 'BN')      
config.set('model_parameters', 'running_stats', 'True')  

config.add_section('dataloader_parameters')
config.set('dataloader_parameters', 'cnn_input', 'False')
config.set('dataloader_parameters', 'bin_length_per_stack', '10')
config.set('dataloader_parameters', 'constant_count', 'False')
config.set('dataloader_parameters', 'constant_duration', 'True')
config.set('dataloader_parameters', 'camera_views', '[3]')
config.set('dataloader_parameters', 'sequence_length', '100')
config.set('dataloader_parameters', 'step_size', '100')
config.set('dataloader_parameters', 'num_bins', '10')
config.set('dataloader_parameters', 'dataset_path', dataset_path)
config.set('dataloader_parameters', 'P_mat_path', P_mat_path)
config.set('dataloader_parameters', 'ev_representation', 'stacked_hist')

config.add_section('training_parameters')
config.set('training_parameters', 'num_epochs', '200')
config.set('training_parameters', 'lr', '0.0001')
config.set('training_parameters', 'batch_size', '2')
config.set('training_parameters', 'num_workers', '4')
config.set('training_parameters', 'num_bins', '10')

config.add_section('inference_parameters')
config.set('inference_parameters', 'batch_size', '4')
config.set('inference_parameters', 'num_workers', '4')
config.set('inference_parameters', 'num_bins', '10')

with open(save_root_dir + r"rnn_config.ini", 'w') as configfile:
    config.write(configfile)

config = configparser.ConfigParser()

config.add_section('model_parameters')
config.set('model_parameters', 'activation_type', 'lif_init')
config.set('model_parameters', 'v_threshold', '1.0')
config.set('model_parameters', 'tau', '3.0')
config.set('model_parameters', 'detach_reset', 'False')
config.set('model_parameters', 'init_mp_dicts', 'False')
config.set('model_parameters', 'output_decay', '0.8') 

config.add_section('dataloader_parameters')
config.set('dataloader_parameters', 'camera_views', '[3,2]')
config.set('dataloader_parameters', 'sequence_length', '100')
config.set('dataloader_parameters', 'step_size', '100')
config.set('dataloader_parameters', 'bin_length_per_stack', '10')
config.set('dataloader_parameters', 'cnn_input', 'True')
config.set('dataloader_parameters', 'constant_count', 'True')
config.set('dataloader_parameters', 'constant_duration', 'False')
config.set('dataloader_parameters', 'dataset_path', dataset_path)
config.set('dataloader_parameters', 'P_mat_path', P_mat_path)
config.set('dataloader_parameters', 'downsample', 'False')
config.set('dataloader_parameters', 'ev_representation', 'stacked_hist')

config.add_section('pretrained_weights')
config.set('pretrained_weights', 'pretrained_weights_camview2', checkpoint_dir + 'HYBRID_CNN_SNN_CONST_COUNT_10x10=100ms_tau3.0_output_decay0.8_camview2_6.pt')
config.set('pretrained_weights', 'pretrained_weights_camview3', checkpoint_dir + 'HYBRID_CNN_SNN_CONST_COUNT_10x10=100ms_tau3.0_output_decay0.8_camview3_2.pt')

config.add_section('inference_parameters')
config.set('inference_parameters', 'num_bins', '10')

with open(save_root_dir + r"triangulation.ini", 'w') as configfile:
    config.write(configfile)