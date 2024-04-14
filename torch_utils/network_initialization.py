from pathlib import Path
import torch.nn as nn
import torch
import pdb

def snn2hybrid_key_conversion(k):
    '''
    Renames pure SNN layers to match layer names in CNN-SNN hybrid model.
    '''
    if 'static_conv.0' in k:
        new_k = k.replace('static_conv.0', 'static_conv_snn.conv2d')
    elif 'static_conv.1' in k:
        new_k = k.replace('static_conv.1', 'static_conv_snn.norm_layer')
    elif 'down1' in k:
        new_k = k.replace('down1', 'down1_snn')
    elif 'down2' in k:
        new_k = k.replace('down2', 'down2_snn')
    elif 'down3' in k:
        new_k = k.replace('down3', 'down3_snn')
    elif 'residualBlock.0' in k: 
        new_k = k.replace('residualBlock.0', 'residualBlock_snn')
    elif 'up1' in k:
        new_k = k.replace('up1', 'up1_snn')
    elif 'up2' in k:
        new_k = k.replace('up2', 'up2_snn')
    elif 'up3' in k:
        new_k = k.replace('up3', 'up3_snn')
    elif 'temporalflat' in k:
        new_k = k.replace('temporalflat', 'temporalflat_snn')
    return new_k

def cnn2hybrid_key_conversion(k):
    '''
    Renames pure CNN layers to match layer names in CNN-SNN hybrid model. 
    '''
    if 'down1' in k:
        new_k = k.replace('down1', 'down1_cnn')
    elif 'down2' in k:
        new_k = k.replace('down2', 'down2_cnn')
    elif 'down3' in k:
        new_k = k.replace('down3', 'down3_cnn')
    elif 'down4' in k:
        new_k = k.replace('down4', 'down4_cnn')
    elif 'down5' in k:
        new_k = k.replace('down5', 'down5_cnn')
    elif 'up1' in k:
        new_k = k.replace('up1', 'up1_cnn')
    elif 'up2' in k:
        new_k = k.replace('up2', 'up2_cnn')
    elif 'up3' in k:
        new_k = k.replace('up3', 'up3_cnn')
    elif 'up4' in k:
        new_k = k.replace('up4', 'up4_cnn')
    elif 'up5' in k:
        new_k = k.replace('up5', 'up5_cnn')
    elif 'conv3' in k:
        new_k = k.replace('conv3', 'conv3_cnn')
    elif 'conv1' in k:
        new_k = k.replace('conv1', 'conv1_cnn')
    elif 'conv2' in k:
        new_k = k.replace('conv2', 'conv2_cnn')
    return new_k

def freeze_cnn_weights(m):
    '''
    Freezes CNN layers for hybrid CNN-SNN training.
    '''
    for name, param in m.named_parameters():
        if 'cnn' in name:
            param.requires_grad = False

def rename_snn_model_params(snn_param_path: str, device):
    '''
    Renames pure SNN layers to match names of hybrid model layers. 

    :snn_param_path: directory of SNN parameters to be loaded to hybrid model. 

    :return: renamed dictionaries for SNN parameters.
    '''

    assert Path(snn_param_path).exists()

    snn_pretrained_dict = torch.load(snn_param_path, map_location=device)
    snn_final_dict = {snn2hybrid_key_conversion(k): snn_pretrained_dict[k] for k in snn_pretrained_dict.keys()}
    
    return snn_final_dict

def rename_cnn_model_params(cnn_param_path: str, device):
    '''
    Renames pure CNN layers to match names of hybrid model layers. 

    :cnn_param_path: directory of CNN parameters to be loaded to hybrid model.

    :return: renamed dictionaries for CNN parameters.
    '''

    assert Path(cnn_param_path).exists()

    cnn_pretrained_dict = torch.load(cnn_param_path, map_location=device)
    cnn_final_dict = {cnn2hybrid_key_conversion(k): cnn_pretrained_dict[k] for k in cnn_pretrained_dict.keys()}

    return cnn_final_dict

def init_weights_div_last_layer(m):
    '''
    Rescales last layer of weights for faster convergence. 
    '''
    div_factor=2500
    if isinstance(m, nn.Conv2d or nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        # if (m.in_channels==64) and (m.out_channels == 13): # for snn
        if (m.in_channels==32) and (m.out_channels == 13): # for cnn
            with torch.no_grad():
                print('Last layer divided.')
                m.weight /= div_factor 


def init_weights(m):
    if isinstance(m, nn.Conv2d or nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)

def model_weights_without_activations(param_path):
    '''
    Input: path of the parameter to be loaded. 
    Return: parameters without activation functions.
    '''
    pretrained_dict = torch.load(param_path)
    final_dict = {k: pretrained_dict[k] for k in pretrained_dict.keys() if k[-2:] != '.w'}
    return final_dict