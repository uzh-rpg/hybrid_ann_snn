from  pathlib import Path 
import argparse 
import configparser
import os

# todo fix bool args 
def add_io_args(parser):
    parser.add_argument('--config_path', help='Config file for remaining parameters.', type=str)
    parser.add_argument('--device', help='Set device.', type=str, default='cpu')     
    return parser

def add_training_args(parser):
    parser.add_argument('--save_params', help='Saving model parameters at every epoch.', action='store_true')
    parser.add_argument('--no-save_params', dest='save_params', action='store_false')
    parser.set_defaults(save_params=True) 
    
    parser.add_argument('--save_runs', help='Plotting experiment to Tensorboard.', action='store_true')
    parser.add_argument('--no-save_runs', dest='save_runs', action='store_false')
    parser.set_defaults(save_runs=True) 
    
    parser.add_argument('--output_path', help='Directory of checkpoints to be saved in if save_params == True.', type=str) 

    parser.add_argument('--init_with_pretrained', help='Initialize with pretrained weights or Xavier weights.', action='store_true')
    parser.add_argument('--no-init_with_pretrained', dest='init_with_pretrained', action='store_false')
    parser.set_defaults(init_with_pretrained=False)

    parser.add_argument('--tb_name', help='Tensorboard name if save_runs == True', type=str) 
    
    return parser 

def add_inference_args(parser):
    parser.add_argument('--pretrained_dict_path', help='Path of pretrained weights.', type=str)
    parser.add_argument('--txt_path', help='Name of txt file for results to be saved.', type=str)
    return parser 

def FLAGS(inference=False):

    parser = argparse.ArgumentParser()
    parser = add_io_args(parser)
    
    if inference: 
        parser = add_inference_args(parser)
    else:
        parser = add_training_args(parser)

    flags = parser.parse_args()
    
    if inference: 
        # pdb.set_trace()
        assert Path(flags.pretrained_dict_path).exists()
    else:
        if not Path(flags.output_path).exists():
            os.mkdir(flags.output_path)

    configfile = configparser.ConfigParser()
    configfile.read(flags.config_path) # todo take from args

    return flags, configfile

