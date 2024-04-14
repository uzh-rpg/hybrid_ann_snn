import torch
import torch.nn.functional as F
from torch import nn
import pdb

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
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class UNet(nn.Module):
    """Modified version of Unet from SuperSloMo.
    """
    # 2 + 10 + 10 + 1 = 23 layers
    def __init__(self, kwargs):
        super(UNet, self).__init__()

        inChannels = kwargs.getint('in_channels')
        outChannels = 13 # changed channel size
        
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        # down: avg pool -> con1 -> leaky relu -> conv2 -> leaky relu (2 layers in tot)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        # up: interpolate -> conv1 -> leaky relu -> conv2 -> leaky relu (2 layers in tot)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)

        x = self.conv3(x)
        return x

if __name__=='__main__':
    from torchsummary import summary
    image_h, image_w, num_joints = 260, 346, 13 
    model = UNet(2)
    summary(model, (2, 256, 256), device='cpu')
