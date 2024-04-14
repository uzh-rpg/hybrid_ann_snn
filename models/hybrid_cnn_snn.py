import pdb
import torch
import os

import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from models.neurons import *

class BasicModel(nn.Module):
    '''
    Basic model class that can be saved and loaded
        with specified names.
    '''

    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print('save model to \"{}\"'.format(path))

    def load(self, path: str):
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)
            self.load_state_dict(state)
            print('load pre-trained model \"{}\"'.format(path))
        else:
            print('init model')
        return self

    def to(self, device: torch.device):
        self.device = device
        super().to(device)
        return self

### START OF SPIKING CLASSES ###
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, tau=2, v_threshold=1.0, v_reset=None, activation_type='lif_init', surrogate_function=surrogate.ATan(), detach_reset=False):
        super(ConvLayer, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation_type == 'lif_init':
            self.activation = CustomVinitLIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        else: 
            assert NotImplementedError
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x, v_init):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        out, mem_pot = self.activation(out, v_init)
        return out, mem_pot

class Spike_recurrentConvLayer_nolstm(nn.Module):
    def __init__(self, in_channels, out_channels, surrogate_function, detach_reset, kernel_size=3, stride=1, padding=0, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Spike_recurrentConvLayer_nolstm, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, tau, v_threshold, v_reset, activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)

    def forward(self, x, v_init):
        x = self.conv(x, v_init)
        return x

class Spiking_residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, surrogate_function, detach_reset, stride=1, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif_init'):
        super(Spiking_residualBlock, self).__init__()
        bias = False
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if activation_type == 'lif_init':
            self.activation1 = CustomVinitLIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
            self.activation2 = CustomVinitLIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        else: 
            assert NotImplementedError
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, v_init):
        residual = x
        out = self.conv1(x)

        out = self.bn1(out)
        out, mem_pot1 = self.activation1(out, v_init[0])
        mid_res = out

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out, mem_pot2= self.activation2(out, v_init[1])
        return out, mid_res, mem_pot1, mem_pot2

class Spike_upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, surrogate_function, detach_reset, stride=1, padding=0, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif_init'):
        super(Spike_upsample_layer, self).__init__()
       
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if activation_type == 'lif_init':
            self.activation = CustomVinitLIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        else: 
            assert NotImplementedError

        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x, v_init):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        out, mem_pot = self.activation(out, v_init)

        return out, mem_pot

class TemporalFlatLayer_concat(nn.Module):
    def __init__(self, surrogate_function, detach_reset, tau=2.0, v_reset=None):
        super(TemporalFlatLayer_concat, self).__init__()

        # self.mp_activation_type = mp_activation_type

        self.conv2d = nn.Conv2d(64, 13, 1, bias=False)
        # if mp_activation_type == 'mp_lif_init':
        #     self.norm_layer = nn.BatchNorm2d(13)
        #     self.activation = CustomVinitMpLIFNode(v_threshold=float('Inf'), tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)

    def forward(self, x, v_init):
        out = self.conv2d(x)

        # if self.mp_activation_type == mp_lif_init:
        #     out = self.norm_layer(out)
        #     out = self.activation(out, v_init)

        return out

### END OF SPIKING CLASSES ###

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

        self.norm_layer = nn.BatchNorm2d(out_channels)
        self.norm_layer2 = nn.BatchNorm2d(out_channels)

    # conv+bn+leakyrelu+conv
    def forward(self, x):
        if self.upsize:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        x = self.conv2(x)
        x = self.norm_layer2(x)

        return x

class Hybrid_CNN_SNN(BasicModel):

    def __init__(self, kwargs = {}, surrogate_function = surrogate.ATan(), inference=False):
        super().__init__()
        
        cnn_in_channels = kwargs.getint('cnn_in_channels')
        activation_type = kwargs['activation_type'] # LIF neuron types
        v_threshold = kwargs.getfloat('v_threshold') # neuron thresholds 
        # v_reset = kwargs['v_reset'] # resting potential of neurons
        tau = kwargs.getfloat('tau') # membrane time constant of neurons
        detach_reset =  kwargs.getboolean('detach_reset') # detach resets for faster training. -- have not been tested. --
        
        self.init_mp_dicts = kwargs.getboolean('init_mp_dicts')
        self.output_decay = kwargs.getfloat('output_decay')

        self.inference = inference

        self.final_output = 0

        ### SNN ###
        self.static_conv_snn = ConvLayer(in_channels = 2, out_channels=32, kernel_size=5, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type=activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)

        self.down1_snn = Spike_recurrentConvLayer_nolstm(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.down2_snn = Spike_recurrentConvLayer_nolstm(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.down3_snn = Spike_recurrentConvLayer_nolstm(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)

        self.residualBlock_snn = Spiking_residualBlock(256, 256, stride=1, tau=tau, v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate_function, detach_reset=detach_reset)

        self.up1_snn = Spike_upsample_layer(in_channels=512, out_channels=128, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.up2_snn = Spike_upsample_layer(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.up3_snn = Spike_upsample_layer(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)

        self.temporalflat_snn = TemporalFlatLayer_concat(tau = tau, v_reset=None, surrogate_function=surrogate_function, detach_reset=detach_reset)

        ### CNN ###
        self.conv1_cnn = nn.Conv2d(cnn_in_channels, 32, 7, stride=1, padding=3)
        self.conv2_cnn = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        # initialize snn first layer neurons 
        self.states_in = state_initializer(32, 32)

        # down: avg pool -> con1 -> leaky relu -> conv2 -> leaky relu (2 layers in tot)
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
        self.states_4 = state_initializer(512, 256)
        
        self.up2_cnn = up(512, 256)
        self.states_5 = state_initializer(256, 256)

        self.up3_cnn = up(256, 128)
        self.states_6 = state_initializer(128, 128)

        self.up4_cnn = up(128, 64)
        self.states_7 = state_initializer(64, 64)

        self.up5_cnn = up(64, 32)
        self.states_8 = state_initializer(32, 32)

        self.conv3_cnn = nn.Conv2d(32, 13, 3, stride=1, padding=1)
        # self.states_out = state_initializer(13, 13) # todo states of the prediction layer

    def forward(self, x_snn=None, x_cnn=None, v_init=False): # 10 layers

        if v_init:
            x_cnn_temp = F.leaky_relu(self.conv1_cnn(x_cnn), negative_slope=0.1)
            x_in_cnn = F.leaky_relu(self.conv2_cnn(x_cnn_temp), negative_slope=0.1)
            v_init_in = self.states_in(x_in_cnn)

            x1_cnn = self.down1_cnn(x_in_cnn)
            v_init_1 = self.states_1(x1_cnn)

            x2_cnn = self.down2_cnn(x1_cnn)
            v_init_2 = self.states_2(x2_cnn)
            
            x3_cnn = self.down3_cnn(x2_cnn)
            v_init_3 = self.states_3(x3_cnn)
        
            x4_cnn = self.down4_cnn(x3_cnn)
            x5_cnn = self.down5_cnn(x4_cnn)
            r1_cnn = self.up1_cnn(x5_cnn, x4_cnn)
            v_init_4 = self.states_4(r1_cnn)

            r2_cnn = self.up2_cnn(r1_cnn, x3_cnn)
            v_init_5 = self.states_5(r2_cnn)

            v_init_r = [v_init_4, v_init_5]

            u1_cnn = self.up3_cnn(r2_cnn, x2_cnn)
            v_init_6 = self.states_6(u1_cnn)

            u2_cnn = self.up4_cnn(u1_cnn, x1_cnn)
            v_init_7 = self.states_7(u2_cnn)
            
            u3_cnn = self.up5_cnn(u2_cnn, x_in_cnn)
            v_init_8 = self.states_8(u3_cnn)

            out_cnn = self.conv3_cnn(u3_cnn)
            # v_init_out = self.states_out(out_cnn)
            v_init_out = None 

            # v_init_in = None
            # if self.feedback and self.two_head:
            #     v_init_in_fb = None
            # v_init_1 = None
            # v_init_2 = None
            # v_init_3 = None
            # v_init_r = [None]*2
            # v_init_6 = None
            # v_init_7 = None
            # v_init_8 = None

            init_mp_vals = {'v_in': v_init_in, 
            'v_1': v_init_1,
            'v_2': v_init_2,
            'v_3': v_init_3,
            'v_4': v_init_4,
            'v_5': v_init_5,
            'v_6': v_init_6,
            'v_7': v_init_7,
            'v_8': v_init_8}

            self.final_output = out_cnn.detach().clone()
            if x_snn == None:
                return init_mp_vals
        else: 
            v_init_in = None
            v_init_1 = None
            v_init_2 = None
            v_init_3 = None
            v_init_r = [None]*2
            v_init_6 = None
            v_init_7 = None
            v_init_8 = None
            v_init_out = None 

        x_in_snn, x_in_mem_pot = self.static_conv_snn(x_snn, v_init=v_init_in)

        x1_snn, x1_mem_pot = self.down1_snn(x_in_snn, v_init=v_init_1)
        x2_snn, x2_mem_pot = self.down2_snn(x1_snn, v_init=v_init_2)
        x3_snn, x3_mem_pot = self.down3_snn(x2_snn, v_init=v_init_3)

        r1_snn, mid_res, r1_mem_pot, r2_mem_pot = self.residualBlock_snn(x3_snn, v_init = v_init_r) # 2 conv layers

        u1_snn, u1_mem_pot = self.up1_snn(torch.cat([r1_snn, x3_snn], 1), v_init=v_init_6)
        u2_snn, u2_mem_pot = self.up2_snn(torch.cat([u1_snn, x2_snn], 1), v_init=v_init_7)
        u3_snn, u3_mem_pot = self.up3_snn(torch.cat([u2_snn, x1_snn], 1), v_init=v_init_8)

        out_snn = self.temporalflat_snn(torch.cat([u3_snn, x_in_snn], 1), v_init=v_init_out)
        
        mp_vals = {'v_in': x_in_mem_pot, 
            'v_1': x1_mem_pot,
            'v_2': x2_mem_pot,
            'v_3': x3_mem_pot,
            'v_4': r1_mem_pot,
            'v_5': r2_mem_pot,
            'v_6': u1_mem_pot,
            'v_7': u2_mem_pot,
            'v_8': u3_mem_pot}

        self.final_output = self.output_decay*self.final_output + out_snn.clone()
        # pdb.set_trace()
        
        if self.inference and v_init:   
            return self.final_output, out_cnn, mid_res
        elif self.inference:
            return self.final_output, mid_res
        elif v_init and self.init_mp_dicts:
            init_mp_vals = {'v_init_in': v_init_in, 
            'v_init_1': v_init_1,
            'v_init_2': v_init_2,
            'v_init_3': v_init_3,
            'v_init_4': v_init_4,
            'v_init_5': v_init_5,
            'v_init_6': v_init_6,
            'v_init_7': v_init_7,
            'v_init_8': v_init_8,
            'v_init_out': v_init_out}      
            return self.final_output, out_cnn, init_mp_vals
        elif v_init:
            return self.final_output, out_cnn
        else: 
            return self.final_output, mp_vals


