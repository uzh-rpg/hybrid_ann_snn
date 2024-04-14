#!/usr/bin/env python
"""
Task dataset implementations.
Provided:
- classification: frame + label
- heatmap: frame + 2d gaussian heatmap
- joints: frame + projected 2d joints
- 3djoints: frame + 3d joints + camera information
- autoencoder: frame

"""
from experimenting.utils.visualization import plot_2d_from_3d
import pdb
import albumentations
import torch
from kornia import geometry
from torch.utils.data import Dataset

from experimenting.utils import Skeleton

from .core import BaseCore

__all__ = [
    "ClassificationDataset",
    "HeatmapDataset",
    "JointsDataset",
    "Joints3DDataset",
    "Joints3DStereoDataset",
    "AutoEncoderDataset",
]

__author__ = "Gianluca Scarpellini"
__license__ = "GPLv3"
__email__ = "gianluca@scarpellini.dev"


class BaseDataset(Dataset):
    def __init__(
        self, dataset: BaseCore, indexes=None, transform=None, augment_label=False
    ):
        self.dataset = dataset
        self.x_indexes = indexes
        self.transform = transform
        self.augment_label = augment_label

    def __len__(self):
        return len(self.dataset.file_paths)

    def _get_x(self, idx):
        """
        Basic get_x utility. Load frame from dataset given frame index

        Args:
          idx: Input index (from 0 to len(self))

        Returns:
          Input loaded from dataset
        """
        x = self.dataset.get_frame_from_id(idx)
        return x

    def _get_y(self, idx):
        pass

    def __getitem__(self, idx):
        # idx = self.x_indexes[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)
        y = self._get_y(idx)

        if self.transform:
            if self.augment_label:
                augmented = self.transform(image=x, mask=y)
                x = augmented["image"]
                y = augmented["mask"]
                y = torch.squeeze(y.transpose(0, -1))
            else:
                augmented = self.transform(image=x)
                x = augmented["image"]
        return x, y


class ClassificationDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):
        super(ClassificationDataset, self).__init__(dataset, indexes, transform, False)

    def _get_y(self, idx):
        return self.dataset.get_label_from_id(idx)


class AutoEncoderDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):
        super(AutoEncoderDataset, self).__init__(dataset, indexes, transform, False)

    def __getitem__(self, idx):
        idx = self.x_indexes[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)

        if self.transform:
            augmented = self.transform(image=x)
            x = augmented["image"]
        return x


class HeatmapDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        super(HeatmapDataset, self).__init__(dataset, indexes, transform, True)

    def _get_y(self, idx):
        return self.dataset.get_heatmap_from_id(idx)


class JointsDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        super(JointsDataset, self).__init__(
            dataset, indexes, transform, augment_label=False
        )

        self.max_h = dataset.MAX_HEIGHT
        self.max_w = dataset.MAX_WIDTH

    def _get_y(self, idx):
        joints_file = self.dataset.get_joint_from_id(idx)

        joints = torch.tensor(joints_file["joints"])
        mask = torch.tensor(joints_file["mask"]).type(torch.bool)
        return (
            geometry.normalize_pixel_coordinates(joints, self.max_h, self.max_w),
            mask,
        )

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self._get_x(idx)
        y, mask = self._get_y(idx)

        if self.transform:
            augmented = self.transform(image=x)
            x = augmented["image"]

        return x, y, mask


class Joints3DDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        super(Joints3DDataset, self).__init__(dataset, indexes, transform, False)

        self.n_joints = dataset.N_JOINTS
        self.height = dataset.in_shape[0]
        self.width = dataset.in_shape[1]

    def _get_y(self, idx):

        sk, intrinsic_matrix, extrinsic_matrix = self.dataset.get_joint_from_id(idx)
        # print(sk)
        sk_onto_cam = sk.project_onto_camera(extrinsic_matrix)

        mask = sk_onto_cam.get_mask()
        sk_onto_cam = sk_onto_cam.get_masked_skeleton(mask)

        sk_normalized = sk_onto_cam.normalize(self.height, self.width, intrinsic_matrix)
        sk_normalized = sk_normalized.get_masked_skeleton(mask)

        joints_2d = sk.get_2d_points(
            self.height,
            self.width,
            extrinsic_matrix=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
        )

        # plot_2d_from_3d(x, gt_skeleton=sk, p_mat=extrinsic_matrix, idx=idx)

        label = {
            "xyz": sk._get_tensor(),
            "skeleton": sk_onto_cam._get_tensor(),
            "normalized_skeleton": sk_normalized._get_tensor(),
            "z_ref": sk_onto_cam.get_z_ref(),
            "2d_joints": joints_2d,
            "M": extrinsic_matrix,
            "camera": intrinsic_matrix,
            "mask": mask,
        }
        return label


class Joints3DStereoDataset(BaseDataset):
    def __init__(self, dataset, indexes=None, transform=None):

        super(Joints3DStereoDataset, self).__init__(dataset, indexes, transform, False)

        self.n_joints = dataset.N_JOINTS
        self.height = dataset.in_shape[0]
        self.width = dataset.in_shape[1]

    def _get_x(self, idx):

        xl = self.dataset.get_frame_from_id(idx[0])
        xr = self.dataset.get_frame_from_id(idx[1])
        return (xl, xr)

    def _get_y(self, idx):
        sk, intrinsic_matrix, extrinsic_matrix = self.dataset.get_joint_from_id(idx)

        sk_onto_cam = sk.project_onto_camera(extrinsic_matrix)

        mask = sk_onto_cam.get_mask()
        sk_onto_cam = sk_onto_cam.get_masked_skeleton(mask)

        sk_normalized = sk_onto_cam.normalize(self.height, self.width, intrinsic_matrix)
        sk_normalized = sk_normalized.get_masked_skeleton(mask)

        joints_2d = sk.get_2d_points(
            self.height,
            self.width,
            extrinsic_matrix=extrinsic_matrix,
            intrinsic_matrix=intrinsic_matrix,
        )

        label = {
            "xyz": sk._get_tensor(),
            "skeleton": sk_onto_cam._get_tensor(),
            "normalized_skeleton": sk_normalized._get_tensor(),
            "z_ref": sk_onto_cam.get_z_ref(),
            "2d_joints": joints_2d,
            "M": extrinsic_matrix,
            "camera": intrinsic_matrix,
            "mask": mask,
        }
        return label
