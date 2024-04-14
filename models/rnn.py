import torch.nn as nn
import torch
import logging

from models.rnn_unet import UNet, UNetRecurrent
from os.path import join
from models.rnn_submodules import ConvLSTM, ResidualBlock, ConvLayer, UpsampleConvLayer, TransposedConvLayer

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        print(config)

        assert('in_channels' in config)
        self.num_bins = config.getint('in_channels')  # number of bins in the voxel grid event tensor

        try:
            self.skip_type = config['skip_type']
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = config.getint('num_encoders')
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = config.getint('base_num_channels')
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = config.getint('num_residual_blocks')
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = config['norm']
        except KeyError:
            self.norm = None

        try:
            self.running_stats = config.getboolean('running_stats')
        except KeyError:
            self.running_stats = True

        try:
            self.use_upsample_conv = config.getboolean('use_upsample_conv')
        except KeyError:
            self.use_upsample_conv = True
        
        try: 
            self.out_channels = config.getint('out_channels')
        except KeyError: 
            self.out_channels = 1

        try: 
            self.activation = config['activation']
        except KeyError: 
            self.activation = 'sigmoid'



class E2VID(BaseE2VID):
    def __init__(self, config):
        super(E2VID, self).__init__(config)

        self.unet = UNet(num_input_channels=self.num_bins,
                         num_output_channels=self.out_channels,
                         skip_type=self.skip_type,
                         activation=self.activation,
                         num_encoders=self.num_encoders,
                         base_num_channels=self.base_num_channels,
                         num_residual_blocks=self.num_residual_blocks,
                         norm=self.norm,
                         running_stats=self.running_stats, 
                         use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor, prev_states=None):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        return self.unet.forward(event_tensor), None


class E2VIDRecurrent(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, config):
        super(E2VIDRecurrent, self).__init__(config)

        try:
            self.recurrent_block_type = config['recurrent_block_type']
        except KeyError:
            self.recurrent_block_type = 'convlstm'  # or 'convgru'

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

    def forward(self, event_tensor, prev_states):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
        img_pred, states = self.unetrecurrent.forward(event_tensor, prev_states)
        return img_pred, states


if __name__=='__main__':
    from torchsummary import summary
    from torch.optim import Adam
    from torch_impl.torch_utils.utils import *

    image_h, image_w, num_joints = 260, 346, 13  # depend on how accumulated frames are generated in Matlab
    # 288 or 256
    # 352 or 320
    model_config = {'num_bins': 2, 
            'out_channels': 13, 
            'skip_type': 'sum', 
            'activation': 'leaky_relu',
            'recurrent_block_type': 'convlstm', 
            'num_encoders': 3, 
            'base_num_channels': 32, 
            'num_residual_blocks': 2, 
            'use_upsample_conv': False, 
            'norm': 'BN',
            'running_stats': True}

    device = 'cpu'
    model = E2VID(model_config).to(device)
    # prev_states = [None] * 3 # times number of encoders
    # print(prev_states)
    summary(model, (2, 256, 256), device='cpu')

    inputs = torch.zeros((2,2,256,256))
    model = E2VIDRecurrent(model_config).to(device)
    model(inputs, None)
    print(model)

    # for name, params in model.named_parameters():
    #     print(name)

    # inputs = torch.zeros((1,2,256,256), device=device)
    # prev_states = None
    # for j in range(10):
    # make_model_timer(model(inputs.float(), prev_states=prev_states))
    # output, states = make_model_timer([model(inputs.float(), prev_states=prev_states) for _ in range(10)])
        # prev_states=states

    # summary(model, (2, 256, 256), device='cpu')
    # inputs = torch.zeros((1,10,2,256,256), device=device)
    # labels = torch.zeros((1,10,13,256,256), device='cuda:0')

    # optimizer = Adam(model.parameters(), lr=1e-3)

    # for j in range(100):
    #     output, states = model(inputs.float(), prev_states=prev_states)

        
    #     print(j)

    #     optimizer.zero_grad()

    #     loss = 50
    #     prev_states = None
    #     for i in range(10):
    #         output, states = model(inputs[:,i].float(), prev_states=prev_states)
    #         loss += my_mse_loss(output, labels[:, i].float(), device='cuda:0')/10
    #         prev_states = states

    #     # backward + optimize + scheduler
    #     with CudaTimer(device=torch.device('cuda:0'), timer_name='loss_backward'):
    #         loss.backward()
    #     with CudaTimer(device=torch.device('cuda:0'), timer_name='optimizer_step'):
    #         optimizer.step()
    # print(model)

    # E2VID RECURRENT 
    # conv2d(2,32,5) Relu # head has no BN layer 
    # encoder0 conv2d(32,64,5) BN(64) Relu ConvLSTM(128,256,3) !!! ConvLSTM dimensions don't match !!!
    # encoder1 conv2d(64,128,5) BN(128) Relu ConvLSTM(256,512,3) 
    # encoder2 conv2d(128,256,5) BN(256) Relu ConvLSTM(512,1024,3)
    # resblock0 conv2d(256,256,3) BN(256) Relu conv2d(256,256,3) BN(256) Relu 
    # resblock1 conv2d(256,256,3) BN(256) Relu conv2d(256,256,3) BN(256) Relu 
    # resblock2 conv2d(256,256,3) BN(256) Relu conv2d(256,256,3) BN(256) Relu 
    # decoder0  Transconv2d(256, 128,5) BN(128) Relu 
    # decoder1 Transconv2d(128,64,5) BN(64) Relu 
    # decoder2 Transconv2d(64,32,5) BN(32) Relu 
    # pred conv2d(32,13,1) BN(13) sigmoid

    # Note summary doesn't print activation layer when the activation layer has been implemented by 
    # 1) calling activation function from a sting using getattr
    # 2) reusing the same activation function (only prints 1st use)

    # for e2vid ConvLSTM is removed



