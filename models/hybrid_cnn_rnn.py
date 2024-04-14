import pdb

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import logging 

from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt 

from models.model_e2vid import *
from models.rnn import BaseE2VID

### START OF CNN CLASSES ###
class up(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
 
    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x

class down(nn.Module):
    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x

class state_initializer(nn.Module):
    def __init__(self, in_channels, out_channels, filterSize=1, last_layer=False):
        super(state_initializer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, filterSize, stride=1, padding=int((filterSize - 1) / 2))

        if in_channels > out_channels:
            self.upsize = True
        else: 
            self.upsize = False
        self.last_layer = last_layer

        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.norm_layer2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.upsize:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.norm_layer2(x)
        return x

class Hybrid_CNN_RNN(BaseE2VID):

    def __init__(self, config):
        super(Hybrid_CNN_RNN, self).__init__(config)

        self.output_tau = config.getfloat('output_decay')

        try:
            self.recurrent_block_type = config['recurrent_block_type']
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

        ### RNN ###
        #header
        self.unetrecurrent = UNetRecurrent(num_input_channels=self.num_bins,
                                           num_output_channels=self.out_channels,
                                           skip_type=self.skip_type,
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation=self.activation,
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           running_stats=self.running_stats, 
                                           use_upsample_conv=self.use_upsample_conv)

        ### CNN ###
        self.conv1_cnn = nn.Conv2d(20, 32, 7, stride=1, padding=3)
        self.conv2_cnn = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        # initialize snn first layer neurons 

        self.down1_cnn = down(32, 64, 5)
        self.states_1 = state_initializer(64, 64)

        self.down2_cnn = down(64, 128, 3)
        self.states_2 = state_initializer(128, 128)

        self.down3_cnn = down(128, 256, 3)
        self.states_3 = state_initializer(256, 256)

        self.down4_cnn = down(256, 512, 3)
        self.down5_cnn = down(512, 512, 3)
        # up: interpolate -> conv1 -> leaky relu -> conv2 -> leaky relu (2 layers in tot)
        self.up1_cnn = up(512, 512)
        
        self.up2_cnn = up(512, 256)

        self.up3_cnn = up(256, 128)

        self.up4_cnn = up(128, 64)

        self.up5_cnn = up(64, 32)

        self.conv3_cnn = nn.Conv2d(32, 13, 3, stride=1, padding=1)


    def forward(self, x_rnn, x_cnn=None, v_init=False, prev_states=None): # 10 layers

        if v_init == True:
            if prev_states != None:
                assert "State initialization can not be done with previous existing states."
            else:
                prev_states = []

            x_cnn_temp = F.leaky_relu(self.conv1_cnn(x_cnn), negative_slope=0.1)
            x_in_cnn = F.leaky_relu(self.conv2_cnn(x_cnn_temp), negative_slope=0.1)

            x1_cnn = self.down1_cnn(x_in_cnn)
            v_init_1_hidden = self.states_1(x1_cnn)
            v_init_1_cell = torch.zeros_like(v_init_1_hidden)
            prev_states.append(tuple([v_init_1_hidden, v_init_1_cell]))

            x2_cnn = self.down2_cnn(x1_cnn)
            v_init_2_hidden = self.states_2(x2_cnn)
            v_init_2_cell = torch.zeros_like(v_init_2_hidden)
            prev_states.append(tuple([v_init_2_hidden, v_init_2_cell]))

            x3_cnn = self.down3_cnn(x2_cnn)
            v_init_3_hidden = self.states_3(x3_cnn)
            v_init_3_cell = torch.zeros_like(v_init_3_hidden)
            prev_states.append(tuple([v_init_3_hidden, v_init_3_cell]))

            x4_cnn = self.down4_cnn(x3_cnn)
            x5_cnn = self.down5_cnn(x4_cnn)
            r1_cnn = self.up1_cnn(x5_cnn, x4_cnn)

            r2_cnn = self.up2_cnn(r1_cnn, x3_cnn)

            u1_cnn = self.up3_cnn(r2_cnn, x2_cnn)

            u2_cnn = self.up4_cnn(u1_cnn, x1_cnn)

            u3_cnn = self.up5_cnn(u2_cnn, x_in_cnn)

            out_cnn = self.conv3_cnn(u3_cnn)

            # final output is the cnn prediction - deatching and cloning the cnn output is not necessary
            self.final_output = out_cnn.detach().clone()

        out_rnn, states = self.unetrecurrent.forward(x_rnn, prev_states)
        
        # snn output is added to final output -  note cloning the snn for summation is not necessary
        self.final_output = self.output_tau*self.final_output + out_rnn.clone()

        if v_init:
            return self.final_output, states, out_cnn
        else:
            return self.final_output, states



