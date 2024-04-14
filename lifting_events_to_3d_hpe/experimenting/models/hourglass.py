from torch import nn

from ..utils import (
    FlatSoftmax,
    _down_stride_block,
    _regular_block,
    _up_stride_block,
    get_feature_extractor,
    init_parameters,
)


class HeatmapPredictor(nn.Module):
    """
    From https://raw.githubusercontent.com/anibali/margipose/
    """
    def __init__(self, n_joints, in_channels):
        super().__init__()
        self.n_joints = n_joints
        self.down_layers = nn.Sequential(
            _regular_block(in_channels, in_channels),
            _regular_block(in_channels, in_channels),
            _down_stride_block(in_channels, 192),
            _regular_block(192, 192),
            _regular_block(192, 192),
        )
        self.up_layers = nn.Sequential(
            _regular_block(192, 192),
            _regular_block(192, 192),
            _up_stride_block(192, in_channels),
            _regular_block(in_channels, in_channels),
            _regular_block(in_channels, self.n_joints),
        )
        init_parameters(self)

    def forward(self, inputs):
        mid_in = self.down_layers(inputs)
        return self.up_layers(mid_in)


class HourglassStage(nn.Module):
    def __init__(self, n_joints, mid_feature_dimension):
        super().__init__()

        self.softmax = FlatSoftmax()

    def forward(self, x):
        out = self.softmax(self.hm_predictor(x))

        return out


class HourglassModel(nn.Module):
    def __init__(self, n_stages, backbone_path, n_joints, n_channels=1):
        super().__init__()

        self.n_stages = n_stages
        self.in_cnn, self.mid_feature_dimension = get_feature_extractor(
            backbone_path)

        self.in_channels = n_channels
        self.softmax = FlatSoftmax()
        self.n_joints = n_joints
        self.hm_combiners = nn.ModuleList()
        self.hg_stages = nn.ModuleList()
        self.softmax = FlatSoftmax()

    class HeatmapCombiner(nn.Module):
        def __init__(self, n_joints, out_channels):
            super().__init__()
            self.combine_block = _regular_block(n_joints, out_channels)

        def forward(self, x):
            return self.combine_block(x)

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    HourglassModel._HeatmapCombiner(
                        self.n_joints, self.mid_feature_dimension))
            self.hg_stages.append(
                HourglassStage(self.n_joints, self.mid_feature_dimension))

    def forward(self, x):
        inp = self.in_cnn(x)

        outs = []
        for t in range(self.n_stages):
            if t > 0:
                inp = inp + self.hm_combiners[t - 1](outs[-1])

            outs.append(self.hg_stages[t](inp))

        return outs
