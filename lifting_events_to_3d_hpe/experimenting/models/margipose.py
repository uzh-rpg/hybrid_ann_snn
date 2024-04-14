"""
Implementation of Margipose Model. Thanks to A.  Nibali --- original source
code: https://github.com/anibali/margipose/src/margipose
"""

from typing import Any, List, Tuple
import pdb
import torch
from torch import nn

# from experimenting.utils.nn_helpers import (    
#     FlatSoftmax,
#     _down_stride_block,
#     _regular_block,
#     _up_stride_block,
#     get_backbone_last_dimension,
#     init_parameters,)
from ..utils import (
    FlatSoftmax,
    _down_stride_block,
    _regular_block,
    _up_stride_block,
    get_backbone_last_dimension,
    init_parameters,
)


class MargiPoseStage(nn.Module):
    MIN_FEATURE_DIMENSION = 64
    MAX_FEATURE_DIMENSION = 128

    def __init__(self, n_joints, mid_shape, heatmap_space, permute):
        super().__init__()

        self.n_joints = n_joints
        self.softmax = FlatSoftmax()
        self.heatmap_space = heatmap_space
        mid_feature_dimension = mid_shape[0]

        up_padding = (
            int(mid_shape[1] % 2 != 1),
            int(mid_shape[2] % 2 != 1),
        )  # TODO: eval up_padding basin on mid_shape
        min_dimension = MargiPoseStage.MIN_FEATURE_DIMENSION
        max_dimension = MargiPoseStage.MAX_FEATURE_DIMENSION

        self.down_layers = nn.Sequential(
            _regular_block(mid_feature_dimension, min_dimension),
            _regular_block(min_dimension, min_dimension),
            _down_stride_block(min_dimension, max_dimension),
            _regular_block(max_dimension, max_dimension),
            _regular_block(max_dimension, max_dimension),
        )
        self.up_layers = nn.Sequential(
            _regular_block(max_dimension, max_dimension),
            _regular_block(max_dimension, max_dimension),
            _up_stride_block(max_dimension, min_dimension, padding=up_padding),
            _regular_block(min_dimension, min_dimension),
            _regular_block(min_dimension, self.n_joints),
        )
        self.permute = permute
        init_parameters(self)

    def forward(self, *inputs):
        mid_in = self.down_layers(inputs[0])
        if self.permute:
            size = mid_in.shape[-1]

            if self.heatmap_space == 'xy':
                mid_out = mid_in
            elif self.heatmap_space == 'zy':
                mid_out = torch.cat(
                    [t.permute(0, 3, 2, 1) for t in mid_in.split(size, -3)], -3
                )
            elif self.heatmap_space == 'xz':

                mid_out = torch.cat(
                    [t.permute(0, 2, 1, 3) for t in mid_in.split(size, -3)], -3
                )
            else:
                raise Exception()
        else:
            mid_out = mid_in
        return self.up_layers(mid_out)


class MargiPoseModel3D(nn.Module):
    """
    Multi-stage marginal heatmap estimator

    """

    def __init__(self, n_stages, in_cnn, in_shape, n_joints, permute_axis=False):
        super().__init__()

        self.n_stages = n_stages
        self.in_cnn = in_cnn  # Backbone provided as parameter

        self.mid_shape = get_backbone_last_dimension(in_cnn, in_shape)
        self.mid_feature_dimension = self.mid_shape[0]

        self.xy_hm_cnns = nn.ModuleList()
        self.zy_hm_cnns = nn.ModuleList()
        self.xz_hm_cnns = nn.ModuleList()
        self.hm_combiners = nn.ModuleList()
        self.softmax = FlatSoftmax()
        self.n_joints = n_joints
        self.permute_axis = permute_axis
        self._set_stages()

    class HeatmapCombiner(nn.Module):
        def __init__(self, n_joints, n_planes, out_channels):
            super().__init__()
            self.combine_block = _regular_block(n_joints * n_planes, out_channels)

        def forward(self, x):
            return self.combine_block(x)

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    MargiPoseModel3D.HeatmapCombiner(
                        self.n_joints, 3, self.mid_feature_dimension
                    )
                )
            self.xy_hm_cnns.append(
                MargiPoseStage(
                    self.n_joints,
                    self.mid_shape,
                    heatmap_space='xy',
                    permute=self.permute_axis,
                )
            )
            self.zy_hm_cnns.append(
                MargiPoseStage(
                    self.n_joints,
                    self.mid_shape,
                    heatmap_space='zy',
                    permute=self.permute_axis,
                )
            )
            self.xz_hm_cnns.append(
                MargiPoseStage(
                    self.n_joints,
                    self.mid_shape,
                    heatmap_space='xz',
                    permute=self.permute_axis,
                )
            )

    def forward(self, inputs) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Model forward process
        Args:
            inputs: input batch

        Returns:
            Triple of list of heatmaps of length {n_stages}
        """
        features = self.in_cnn(inputs)
        # pdb.set_trace()
        xy_heatmaps: List[Any] = []
        zy_heatmaps: List[Any] = []
        xz_heatmaps: List[Any] = []

        inp = features
        # pdb.set_trace()
        # xy_heatmaps0 = self.softmax(self.xy_hm_cnns[0](inp))
        # zy_heatmaps0 = self.softmax(self.zy_hm_cnns[0](inp))
        # xz_heatmaps0 = self.softmax(self.xz_hm_cnns[0](inp))
                
        # combined_hm_features = self.hm_combiners[0](torch.cat([xy_heatmaps0, zy_heatmaps0, xz_heatmaps0], -3))
        
        # inp = inp[:,:,:,:39] + combined_hm_features

        # xy_heatmaps1 = self.softmax(self.xy_hm_cnns[1](inp))
        # zy_heatmaps1 = self.softmax(self.zy_hm_cnns[1](inp))
        # xz_heatmaps1 = self.softmax(self.xz_hm_cnns[1](inp))

        # combined_hm_features = self.hm_combiners[1](torch.cat([xy_heatmaps1, zy_heatmaps1, xz_heatmaps1], -3))
        
        # inp = inp + combined_hm_features

        # xy_heatmaps2 = self.softmax(self.xy_hm_cnns[2](inp))
        # zy_heatmaps2 = self.softmax(self.zy_hm_cnns[2](inp))
        # xz_heatmaps2 = self.softmax(self.xz_hm_cnns[2](inp))

        # # pdb.set_trace()
        for t in range(self.n_stages):
            if t > 0:
                combined_hm_features = self.hm_combiners[t - 1](
                    torch.cat(
                        [xy_heatmaps[t - 1], zy_heatmaps[t - 1], xz_heatmaps[t - 1]], -3
                    )
                )
                inp = inp + combined_hm_features

            xy_heatmaps.append(self.softmax(self.xy_hm_cnns[t](inp)))
            zy_heatmaps.append(self.softmax(self.zy_hm_cnns[t](inp)))
            xz_heatmaps.append(self.softmax(self.xz_hm_cnns[t](inp)))
        
        return xy_heatmaps, zy_heatmaps, xz_heatmaps
        # return xy_heatmaps2, zy_heatmaps2, xz_heatmaps2


class MargiPoseModel2D(MargiPoseModel3D):
    def __init__(self, n_stages, backbone_path, n_joints, n_channels=1):
        super().__init__(n_stages, backbone_path, n_joints, n_channels)

    class HeatmapCombiner(nn.Module):
        def __init__(self, n_joints, n_planes, out_channels):
            super().__init__()
            self.combine_block = _regular_block(n_joints * n_planes, out_channels)

        def forward(self, x):
            return self.combine_block(x)

    def _set_stages(self):
        for t in range(self.n_stages):
            if t > 0:
                self.hm_combiners.append(
                    MargiPoseModel2D.HeatmapCombiner(
                        self.n_joints, 1, self.mid_feature_dimension
                    )
                )
            self.xy_hm_cnns.append(
                MargiPoseStage(
                    self.n_joints,
                    self.mid_feature_dimension,
                    heatmap_space='xy',
                    permute=False,
                )
            )

    def forward(self, inputs):
        features = self.in_cnn(inputs)
        xy_heatmaps = []

        inp = features
        for t in range(self.n_stages):
            if t > 0:
                combined_hm_features = self.hm_combiners[t - 1](xy_heatmaps[t - 1])
                inp = inp + combined_hm_features
            xy_heatmaps.append(self.softmax(self.xy_hm_cnns[t](inp)))

        return xy_heatmaps


def get_margipose_model(params):
    return MargiPoseModel3D(**params)

# if __name__ == '__main__':
#     from torchsummary import summary

#     model = MargiPoseModel3D()
#     summary(model, (1, 256, 256))