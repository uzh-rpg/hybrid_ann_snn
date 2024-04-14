import pdb

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

from models.neurons import *
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pad_sequence

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

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, tau=2, v_threshold=1.0, v_reset=None, activation_type='lif', surrogate_function=surrogate.ATan(), detach_reset=False):
        super(ConvLayer, self).__init__()

        bias = False
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation_type == 'lif':
            self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        elif activation_type == 'if':
            self.activation = IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        elif activation_type == 'plif':
            self.activation = ParametricLIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        out = self.activation(out)
        return out

class Spike_recurrentConvLayer_nolstm(nn.Module):
    def __init__(self, in_channels, out_channels, surrogate_function, detach_reset, kernel_size=3, stride=1, padding=0, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Spike_recurrentConvLayer_nolstm, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, tau, v_threshold, v_reset, activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)

    def forward(self, x):
        x = self.conv(x)
        return x

class Spiking_residualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, surrogate_function, detach_reset, stride=1, tau=2, v_threshold=1.0, v_reset=None):
        super(Spiking_residualBlock, self).__init__()
        bias = False
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # todo create new LIF neuron
        self.lif1 = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.lif2 = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        residual = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.lif1(out)
        mid_res = out

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.lif2(out)
        return out, mid_res

class Spike_upsample_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, surrogate_function, detach_reset, stride=1, padding=0, tau=2, v_threshold=1.0, v_reset=None, activation_type = 'lif'):
        super(Spike_upsample_layer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if activation_type == 'lif':
            self.activation = LIFNode(v_threshold=v_threshold, tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        elif activation_type == 'if':
            self.activation = IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        elif activation_type == 'plif':
            self.activation = ParametricLIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.norm_layer = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)
        out = self.norm_layer(out)
        out = self.activation(out)

        return out

class TemporalFlatLayer_concat(nn.Module):
    def __init__(self, surrogate_function, detach_reset, tau=2.0, v_reset=None):
        super(TemporalFlatLayer_concat, self).__init__()

        self.conv2d = nn.Conv2d(64, 13, 1, bias=False) # change: make output channel size 13
        self.norm_layer = nn.BatchNorm2d(13)
        self.activation = MpLIFNode(v_threshold=float('Inf'), tau = tau, v_reset=v_reset, surrogate_function=surrogate_function, detach_reset=detach_reset)

    def forward(self, x, last_mem):
        out = self.conv2d(x)
        out = self.norm_layer(out)
        out = self.activation(out, last_mem)
        return out

class EVSNN_LIF_final(BasicModel):

    def __init__(self, kwargs = {}, surrogate_function = surrogate.ATan(), inference=False):
        super().__init__()
        activation_type = kwargs['activation_type']
        v_threshold = kwargs.getfloat('v_threshold')
        tau = kwargs.getfloat('tau')
        detach_reset =  kwargs.getboolean('detach_reset')

        self.inference = inference
        in_channels = 2
        #header
        self.static_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            LIFNode(v_threshold=v_threshold, tau = tau, v_reset=None, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )
        self.down1 = Spike_recurrentConvLayer_nolstm(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.down2 = Spike_recurrentConvLayer_nolstm(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.down3 = Spike_recurrentConvLayer_nolstm(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)

        self.residualBlock = nn.Sequential(
            Spiking_residualBlock(256, 256, stride=1, tau=tau, v_threshold=v_threshold, v_reset=None, surrogate_function=surrogate_function, detach_reset=detach_reset),
        )

        self.up1 = Spike_upsample_layer(in_channels=512, out_channels=128, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.up2 = Spike_upsample_layer(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)
        self.up3 = Spike_upsample_layer(in_channels=128, out_channels=32, kernel_size=5, stride=1, padding=2, tau=tau, v_threshold=v_threshold, v_reset=None, activation_type = activation_type, surrogate_function=surrogate_function, detach_reset=detach_reset)

        self.temporalflat = TemporalFlatLayer_concat(tau = tau, v_reset=None, surrogate_function=surrogate_function, detach_reset=detach_reset)

    def forward(self, x, prev_mem_states): # 10 layers
        
        # with CudaTimer(device=torch.device('cuda:0'), timer_name='head'):
        x_in = self.static_conv(x)

        # with CudaTimer(device=torch.device('cuda:0'), timer_name='encoder_0'):
        x1 = self.down1(x_in)
        # with CudaTimer(device=torch.device('cuda:0'), timer_name='encoder_1'):
        x2 = self.down2(x1)
        # with CudaTimer(device=torch.device('cuda:0'), timer_name='encoder_2'):
        x3 = self.down3(x2)

        # with CudaTimer(device=torch.device('cuda:0'), timer_name='resblock'):
        r1, mid_res = self.residualBlock(x3) # 2 conv layers

        # with CudaTimer(device=torch.device('cuda:0'), timer_name='decoder_0'):
        u1 = self.up1(torch.cat([r1, x3], 1))
        # with CudaTimer(device=torch.device('cuda:0'), timer_name='decoder_1'):        
        u2 = self.up2(torch.cat([u1, x2], 1))
        # with CudaTimer(device=torch.device('cuda:0'), timer_name='decoder_2'):
        u3 = self.up3(torch.cat([u2, x1], 1))
        # with CudaTimer(device=torch.device('cuda:0'), timer_name='tail'):
        membrane_potential = self.temporalflat(torch.cat([u3, x_in], 1), prev_mem_states)

        if self.inference:
            return membrane_potential, mid_res
        else:
            return membrane_potential
