"""
Losses implementations
"""

import torch
import torch.nn as nn
from kornia.geometry import spatial_expectation2d

from ..agents.margipose_estimator import predict3d
from ..utils import (
    SoftArgmax2D,
    average_loss,
    get_joints_from_heatmap,
    js_reg_losses,
)
from .metrics import MPJPE

__all__ = ['HeatmapLoss', 'PixelWiseLoss', 'MultiPixelWiseLoss']


class HeatmapLoss(nn.Module):
    """
    from https://github.com/anibali/margipose
    """

    def __init__(self, reduction='mask_mean', n_joints=13):
        """
        Args:
         reduction (String, optional): only "mask" methods allowed
        """
        super(HeatmapLoss, self).__init__()
        self.divergence = js_reg_losses
        self.reduction = _get_reduction(reduction)
        self.soft_argmax = SoftArgmax2D(window_fn="Uniform")
        self.n_joints = n_joints

    def _mpjpe(self, y_pr, y_gt, reduce=False):
        """
        y_pr = heatmap obtained with CNN
        y_gt = 2d points of joints, in order
        """

        p_coords_max = self.soft_argmax(y_pr)
        gt_coords, _ = get_joints_from_heatmap(y_gt)

        dist_2d = torch.norm((gt_coords - p_coords_max), dim=-1)
        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints
            gt_mask = y_gt.view(y_gt.size()[0], -1, self.n_joints).sum(1) > 0
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d

    def forward(self, pred, gt):
        ndims = 2
        n_joints = pred.shape[1]

        loss = torch.add(self._mpjpe(pred, gt), self.divergence(pred, gt, ndims))
        gt_mask = gt.view(gt.size()[0], -1, n_joints).sum(1) > 0

        return self.reduction(loss, gt_mask)


class PixelWiseLoss(nn.Module):
    """
    from https://github.com/anibali/margipose/
    """

    def __init__(self, reduction='mask_mean', divergence=True):
        super(PixelWiseLoss, self).__init__()
        self.divergence = divergence
        self.mpjpe = MPJPE()
        self.sigma = 1
        self.reduction = _get_reduction(reduction)
        self.divergence = divergence

    def forward(self, pred_hm, gt_joints, gt_mask=None):
        if type(pred_hm) == tuple:
            pred_hm = pred_hm[0]
        gt_joints = gt_joints.narrow(-1, 0, 2)
        pred_joints = spatial_expectation2d(pred_hm)
        loss = self.mpjpe(pred_joints, gt_joints, gt_mask)
        if self.divergence:
            loss += js_reg_losses(pred_hm, gt_joints, self.sigma)
        return self.reduction(loss, gt_mask)


class MultiPixelWiseLoss(PixelWiseLoss):
    """
    from https://github.com/anibali/margipose
    """

    def __init__(self, reduction='mask_mean', divergence=True):
        """
        Args:
            reduction (String, optional): only "mask" methods allowed
        """

        super(MultiPixelWiseLoss, self).__init__(reduction, divergence)

    def forward(self, pred_hm, gt_joints, gt_mask=None):

        pred_xy_hm, pred_zy_hm, pred_xz_hm = pred_hm

        target_xy = gt_joints.narrow(-1, 0, 2)
        target_zy = torch.cat(
            [gt_joints.narrow(-1, 2, 1), gt_joints.narrow(-1, 1, 1)], -1
        )
        target_xz = torch.cat(
            [gt_joints.narrow(-1, 0, 1), gt_joints.narrow(-1, 2, 1)], -1
        )

        pred_joints = predict3d(pred_xy_hm, pred_zy_hm, pred_xz_hm)

        loss = self.mpjpe(pred_joints, gt_joints, gt_mask)
        if self.divergence:
            loss = loss + js_reg_losses(pred_xy_hm, target_xy, self.sigma)
            loss = loss + js_reg_losses(pred_zy_hm, target_zy, self.sigma)
            loss = loss + js_reg_losses(pred_xz_hm, target_xz, self.sigma)

        result = self.reduction(loss, gt_mask)

        return result


def _get_reduction(reduction_type):
    switch = {'mean': torch.mean, 'mask_mean': average_loss, 'sum': torch.sum}

    return switch[reduction_type]
