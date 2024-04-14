import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import init_parameters, get_backbone_last_dimension


class AutoEncoder(nn.Module):
    def __init__(self, in_shape, in_cnn, latent_size, up_layers=4):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_shape[0]

        mid_dimension = get_backbone_last_dimension(in_cnn, in_shape)
        self.in_cnn = in_cnn

        self.mid_feature_dimension = mid_dimension[0]

        self.latent_size = latent_size
        self.mid_tensor_dimension = mid_dimension
        self.dec_channels = 32
        self.up_layers = up_layers

        self.encoder_head = nn.Linear(
            self.mid_feature_dimension * self.mid_tensor_dimension[1] *
            self.mid_tensor_dimension[2], latent_size)

        ###############
        # DECODER
        ##############

        self.d_fc_1 = nn.Linear(
            latent_size,
            self.dec_channels * (2**up_layers) * self.mid_tensor_dimension[1] *
            self.mid_tensor_dimension[2])

        self.decoders = nn.ModuleList()
        for i in reversed(range(2, up_layers + 1)):
            mid_channels = self.dec_channels * (2**i)
            out_channels = self.dec_channels * (2**(i - 1))
            stage = nn.Sequential(
                nn.Conv2d(mid_channels,
                          out_channels,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=0), nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(out_channels))
            self.decoders.append(stage)

        self.d_conv_final = nn.Conv2d(self.dec_channels * 2,
                                      self.in_channels,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=0)

        self.avg_pool = nn.AdaptiveMaxPool2d(
            (in_shape[1], in_shape[2]))
        init_parameters(self)

    def encode(self, x):
        x = self.in_cnn(x)

        x = x.view(
            -1, self.mid_feature_dimension * self.mid_tensor_dimension[1] *
            self.mid_tensor_dimension[2])
        x = self.encoder_head(x)
        return x

    def decode(self, x):

        in_channels = self.dec_channels * (2**self.up_layers)
        x = self.d_fc_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = x.view(-1, in_channels, self.mid_tensor_dimension[1],
                   self.mid_tensor_dimension[2])

        # uplayers stage
        for i in range(len(self.decoders)):
            x = F.interpolate(x, scale_factor=2)

            x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
            x = self.decoders[i](x)

        # out
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.avg_pool(self.d_conv_final(x))
        x = torch.sigmoid(x)

        return x

    def forward(self, x):

        z = self.encode(x)

        decoded = self.decode(z)
        return decoded
