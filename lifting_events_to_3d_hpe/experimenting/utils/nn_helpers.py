"""
Integration toolbox for pytorch nn package
"""
import pdb
import os
from math import sqrt

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from kornia.geometry import spatial_softmax2d
from torch.nn import init
from torch.nn.modules.conv import _ConvNd
from torchvision import models

from ..models import DHP19Model

__all__ = [
    "FlatSoftmax",
    "get_feature_extractor",
    "_regular_block",
    "_up_stride_block",
    "init_parameters",
    "_down_stride_block",
    "get_cnn",
    "get_backbone_last_dimension",
]


class FlatSoftmax(nn.Module):
    def __init__(self):
        super(FlatSoftmax, self).__init__()

    def forward(self, inp):
        return spatial_softmax2d(inp)


def _regular_block(in_chans, out_chans):
    return ResidualBlock(
        out_chans,
        nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
        nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
    )


def _down_stride_block(in_chans, out_chans, padding=1):
    return ResidualBlock(
        out_chans,
        nn.Conv2d(
            in_chans, out_chans, kernel_size=3, padding=padding, stride=2, bias=False
        ),
        nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2, bias=False),
    )


def _up_stride_block(in_chans, out_chans, padding=(0, 1)):
    return ResidualBlock(
        out_chans,
        nn.ConvTranspose2d(
            in_chans,
            out_chans,
            kernel_size=3,
            padding=1,
            stride=2,
            output_padding=padding,
            bias=False,
        ),
        nn.ConvTranspose2d(
            in_chans,
            out_chans,
            kernel_size=1,
            stride=2,
            output_padding=padding,
            bias=False,
        ),
    )


def init_parameters(net):
    for m in net.modules():
        if isinstance(m, _ConvNd):
            init.kaiming_normal_(m.weight, 0, "fan_out")
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Kaiming initialisation for linear layers
            init.normal_(m.weight, 0, sqrt(2.0 / m.weight.size(0)))
            if m.bias is not None:
                init.normal_(m.bias, 0, sqrt(2.0 / m.bias.size(0)))
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    """
    From https://raw.githubusercontent.com/anibali/margipose/
    """

    def __init__(self, chans, main_conv_in, shortcut_conv_in):
        super().__init__()
        assert main_conv_in.in_channels == shortcut_conv_in.in_channels
        self.module = nn.Sequential(
            main_conv_in,
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans, chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(shortcut_conv_in, nn.BatchNorm2d(chans))

    def forward(self, *inputs):
        return self.module(inputs[0]) + self.shortcut(inputs[0])


def get_feature_extractor(params):
    # pdb.set_trace()
    switch = {
        "resnet34": _get_resnet34_cut_128,
        "resnet50": _get_resnet50_feature_extactor,
        "resnet34_cut_256": _get_resnet34_cut_256,
        "resnet34_cut_512": _get_resnet34_cut_512,
        "default": _load_as_is,
    }

    params["n_classes"] = 1  # just placehodler

    if params["model"] not in switch:
        params["model"] = "default"
    if "custom_model_path" in params and not os.path.exists(
        params["custom_model_path"]
    ):
        params.pop("custom_model_path")
        raise Exception("Custom model not found!")

    return switch.get(params["model"])(params)


def _load_as_is(params):
    net = torch.load(params["custom_model_path"])

    if "in_cnn" in dir(net):
        net = net.in_cnn
    return net


def _load_resnet34(params):
    if "custom_model_path" in params:
        resnet = torch.load(params["custom_model_path"])
    else:
        resnet = get_cnn(
            "resnet34",
            {
                "n_channels": params["n_channels"],
                "pretrained": params["pretrained"],
                "n_classes": params["n_classes"],
            },
        )
    return resnet


def _load_resnet50(params):
    if "custom_model_path" in params:
        resnet = torch.load(params["custom_model_path"])
    else:
        resnet = get_cnn(
            "resnet50",
            {
                "n_channels": params["n_channels"],
                "pretrained": params["pretrained"],
                "n_classes": params["n_classes"],
            },
        )
    return resnet


def _get_resnet34_cut_128(params):

    resnet = _load_resnet34(params)
    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
    )

    return net


def _get_resnet34_cut_256(params):
    resnet = _load_resnet34(params)

    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    )

    return net


def _get_resnet34_cut_512(params):
    resnet = _load_resnet34(params)
    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )

    return net


def _get_resnet50_feature_extactor(params):
    resnet = _load_resnet50(params)

    net = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
    )
    return net


def _get_mobilenetv2(n_channels, n_classes, pretrained=False):
    cnn = models.mobilenet_v2(pretrained)
    if n_channels != 3:
        cnn.features[0][0] = torch.nn.Conv2d(
            n_channels,
            32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
        )

    num_ftrs = cnn.classifier[-1].in_features
    cnn.classifier[-1] = nn.Linear(num_ftrs, n_classes, bias=True)
    return cnn


def _get_resnet(resnet, n_channels, n_classes, pretrained=False):
    cnn = getattr(models, resnet)(pretrained)

    if n_channels != 3:
        cnn.conv1 = torch.nn.Conv2d(
            n_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    num_ftrs = cnn.fc.in_features
    cnn.fc = nn.Linear(num_ftrs, n_classes)

    return cnn


def _get_unet_resnet(resnet, n_channels, n_classes, pretrained=False, encoder_depth=3):
    encoder_weights = "imagenet" if pretrained else None
    encoder_depth = 3
    decoder_channels = tuple(
        [16 * (2 ** i) for i in reversed(range(0, int(encoder_depth)))]
    )
    model = smp.Unet(
        resnet,
        encoder_weights=encoder_weights,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
        classes=n_classes,
        activation=None,
    )

    model.encoder.conv1 = nn.Conv2d(
        n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.segmentation_head[-1] = FlatSoftmax()

    return model


def _get_dhp19_model(n_channels, n_classes):
    return DHP19Model(n_channels, n_classes)


def get_cnn(model_name, params):
    switcher = {
        "resnet18": lambda **args: _get_resnet("resnet18", **args),
        "resnet34": lambda **args: _get_resnet("resnet34", **args),
        "resnet50": lambda **args: _get_resnet("resnet50", **args),
        "unet_resnet18": lambda **args: _get_unet_resnet("resnet18", **args),
        "unet_resnet34": lambda **args: _get_unet_resnet("resnet34", **args),
        "dhp19": _get_dhp19_model,
    }
    return switcher[model_name](**params)


def get_backbone_last_dimension(net, input_shape):
    x = torch.randn((32, *input_shape))
    return net(x).shape[1:]
