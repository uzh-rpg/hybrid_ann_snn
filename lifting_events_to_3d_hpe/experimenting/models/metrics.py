"""
Metrics implementation for 3D human pose comparisons
"""
import pdb
import torch
from torch import nn
import matplotlib.pyplot as plt

from experimenting.utils.cv_helpers import (
    _project_xyz_onto_image,
    compose_projection_matrix,
    ensure_homogeneous,
    project_xyz_onto_camera_coord,
    reproject_xyz_onto_world_coord,
)
from experimenting.utils.visualization import plot_skeleton_2d, plot_skeleton_3d
from experimenting.utils import Skeleton

__all__ = ['MPJPE', 'AUC', 'PCK', 'MPJPE2D']


class BaseMetric(nn.Module):
    def forward(self, y_pr, points_gt, gt_mask=None):
        """
        Base forward method for metric evaluation
        Args:
            y_pr: 3D prediction of joints, tensor of shape (BATCH_SIZExN_JOINTSx3)
            points_gt: 3D gt of joints, tensor of shape (BATCH_SIZExN_JOINTSx3)
            gt_mask: boolean mask, tensor of shape (BATCH_SIZExN_JOINTS). 
            Applied to results, if provided

        Returns:
            Metric as single value, if reduction is given, or as a tensor of values
        """
        pass


class MPJPE(BaseMetric):
    def __init__(self, reduction=None, confidence=0, **kwargs):
        super().__init__(**kwargs)
        self.confidence = confidence
        self.reduction = reduction

    def forward(self, y_pr, points_gt, gt_mask=None):
        if gt_mask is not None:
            points_gt[~gt_mask] = 0

        dist_2d = torch.norm((points_gt - y_pr), dim=-1)

        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt
            # joints
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d

class MPJPE2D(nn.Module):
    def __init__(self, reduction=None, confidence=0, **kwargs):
        super().__init__()
        self.confidence = confidence
        self.reduction = reduction



# pred_sk = Skeleton(preds[0].detach().numpy()).denormalize(260, 346, camera=b_y['camera'][0], z_ref=b_y['z_ref'][0]).reproject_onto_world(b_y['M'][0])
# pred_joints = pred_sk.get_2d_points(260, 346, extrinsic_matrix=b_y['M'][0], intrinsic_matrix=b_y['camera'][0])

    def forward(self, y_pr, points_2d_gt, intrinsic_matrix, extrinsic_matrix, z_ref, gt_mask=None, b_x=None):
        points_2d_pred = torch.zeros_like(points_2d_gt, device='cpu')
        intrinsic_matrix = intrinsic_matrix.cpu()
        extrinsic_matrix = extrinsic_matrix.cpu()
        z_ref = z_ref.cpu()
        # 1) convert 3D keypoints to 2D keypoints
        for i in range(points_2d_gt.shape[0]):
            # pred_sk = Skeleton(y_pr[i].cpu()).denormalize(346, 260, camera=intrinsic_matrix[i], z_ref=z_ref[i]).reproject_onto_world(extrinsic_matrix[i])
            pred_sk = Skeleton(y_pr[i].cpu()).reproject_onto_world(extrinsic_matrix[i])
            # sk_onto_world = sk.reproject_onto_world(extrinsic_matrix[i].cpu())
            temp = pred_sk.get_2d_points(346, 260, extrinsic_matrix=extrinsic_matrix[i], intrinsic_matrix=intrinsic_matrix[i])
            # temp = sk_onto_world.get_2d_points(346, 260, extrinsic_matrix=extrinsic_matrix[i].cpu(), intrinsic_matrix=intrinsic_matrix[i].cpu())
            points_2d_pred[i] = torch.from_numpy(temp)
            # plot_skeleton_2d(b_x[0].squeeze().cpu().numpy(), points_2d_gt[i].cpu().numpy(), points_2d_pred[i].cpu().numpy())
            # plt.savefig('test.png')
            # pdb.set_trace()
        # pdb.set_trace()

            # plot_skeleton_2d(b_x[i,0].cpu().numpy(), points_2d_gt[i].cpu().numpy())
            # plt.savefig('/longtermdatasets/asude/lifting_events_to_3d_hpe/experimenting/models/img{}.png'.format(i))
        points_2d_pred = points_2d_pred.to(points_2d_gt.device)

        # 2) calculate 2D MPJPE 
        if gt_mask is not None:
            points_2d_gt[~gt_mask] = 0

        # pdb.set_trace()

        dist_2d = torch.norm((points_2d_gt - points_2d_pred), dim=-1)
        if self.reduction:
            # To apply a reduction method (e.g. mean) we need a mask of gt joints
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d
    
    def get_2d_points(self, y_pr, height=346, width=260, p_mat=None, extrinsic_matrix=None, intrinsic_matrix=None):
        
        if p_mat is None:
            p_mat = compose_projection_matrix(intrinsic_matrix[:3], extrinsic_matrix)
        points = self._get_tensor()[:, :3].transpose(1, 0)
        xj, yj, mask = _project_xyz_onto_image(points.numpy(), p_mat.numpy(), height, width)
        joints = np.array([xj * mask, yj * mask]).transpose(1, 0)
        return joints

    def _get_tensor(self) -> torch.Tensor:
        return self._skeleton.narrow(-1, 0, 3)


class PCK(BaseMetric):
    """
    Percentage of correct keypoints according to a thresold value. Usually
    default threshold is 150mm
    """
    def __init__(self, reduction=None, threshold=150, **kwargs):
        super().__init__(**kwargs)
        self.thr = threshold
        self.reduction = reduction

    def forward(self, y_pr, points_gt, gt_mask=None):
        if gt_mask is not None:
            points_gt[~gt_mask] = 0

        dist_2d = (torch.norm((points_gt - y_pr), dim=-1) < self.thr).double()
        if self.reduction:
            dist_2d = self.reduction(dist_2d, gt_mask)
        return dist_2d


class AUC(BaseMetric):
    """
    Area Under the Curve for PCK metric, 
    at different thresholds (from 0 to 800)
    """
    def __init__(self,
                 reduction=None,
                 auc_reduction=torch.mean,
                 start_at=0,
                 end_at=500,
                 step=30,
                 **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.auc_reduction = auc_reduction
        self.thresholds = torch.linspace(start_at, end_at, step).tolist()

    def forward(self, y_pr, points_gt, gt_mask=None):
        pck_values = torch.DoubleTensor(len(self.thresholds))
        for i, threshold in enumerate(self.thresholds):
            _pck = PCK(self.reduction, threshold=threshold)
            pck_values[i] = _pck(y_pr, points_gt, gt_mask)

        if self.auc_reduction:
            pck_values = torch.mean(pck_values)

        return pck_values
