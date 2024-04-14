import torch.nn as nn
from kornia.geometry import spatial_softmax2d


class DHP19Model(nn.Module):
    def __init__(self, n_channels, n_joints):
        super(DHP19Model, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=n_channels,
                      out_channels=16,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU())

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU())

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU())
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               kernel_size=3), nn.LeakyReLU())
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=16,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               kernel_size=3), nn.LeakyReLU())

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      padding=1), nn.LeakyReLU())
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=n_joints,
                      kernel_size=3,
                      padding=1))

    def forward(self, x):

        x1 = self.block1(x)
        x = self.max_pool(x1)
        x2 = self.block2(x)
        x = self.max_pool(x2)
        x3 = self.block3(x)

        x4 = self.up1(x3)
        x = x2 + x4
        x5 = self.block4(x)
        x6 = self.up2(x5)
        x = x6 + x1
        x7 = self.block5(x)
        out = self.head(x7)
        return spatial_softmax2d(out)
