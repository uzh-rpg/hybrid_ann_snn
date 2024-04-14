"""
Core dataset implementation. BaseCore may be inherhit to create a new
DatasetCore
"""

import os
from abc import ABC, abstractmethod
import pdb
import cv2
import numpy as np
import torch
from scipy import io

from experimenting.utils import Skeleton

from ..utils import get_file_paths
from .base import BaseCore


class DHP19Core(BaseCore):
    """
    DHP19 dataset core class. It provides implementation to load frames,
    heatmaps, 2D joints, 3D joints
    """

    MOVEMENTS_PER_SESSION = {1: 8, 2: 6, 3: 6, 4: 6, 5: 7}
    MAX_WIDTH = 346
    MAX_HEIGHT = 260
    N_JOINTS = 13
    N_MOVEMENTS = 33
    DEFAULT_TEST_SUBJECTS = [1, 2, 3, 4, 5]
    DEFAULT_TEST_VIEW = [1, 2]
    TORSO_LENGTH = 453.5242317

    def __init__(
        self,
        name,
        data_dir,
        cams,
        movements,
        joints_dir,
        n_classes,
        n_joints,
        partition,
        n_channels,
        test_subjects=None,
        test_cams=None,
        avg_torso_length=TORSO_LENGTH,
        *args,
        **kwargs,
    ):
        super(DHP19Core, self).__init__(name, partition)
        self.file_paths = DHP19Core._get_file_paths_with_cam_and_mov(
            data_dir, cams, movements
        )

        self.in_shape = (DHP19Core.MAX_HEIGHT, DHP19Core.MAX_WIDTH)
        self.n_channels = n_channels
        self.n_joints = n_joints

        self.avg_torso_length = avg_torso_length
        
        self.classification_labels = [
            DHP19Core.get_label_from_filename(x_path) for x_path in self.file_paths
        ]

        self.frames_info = [DHP19Core.get_frame_info(x) for x in self.file_paths]
        self.joints = self._retrieve_data_files(joints_dir, f"_2dhm.npz") # retrieve content of files

        self.test_subjects = test_subjects
        if test_cams is None:
            self.test_cams = DHP19Core.DEFAULT_TEST_VIEW
        else:
            self.test_cams = test_cams


    @staticmethod
    def get_standard_path(subject, session, movement, frame, cam, postfix=""):
        return "S{}_session_{}_mov_{}_frame_{}_cam_{}{}.npy".format(
            subject, session, movement, frame, cam, postfix
        )

    @staticmethod
    def load_frame(path):
        ext = os.path.splitext(path)[1]
        if ext == ".mat":
            x = DHP19Core._load_matlab_frame(path)
        elif ext == ".npy":
            x = np.load(path, allow_pickle=True) / 255.0
            if len(x.shape) == 2:
                x = np.expand_dims(x, -1)
        return x

    @staticmethod
    def _load_matlab_frame(path):
        """
        Matlab files contain voxelgrid frames and must be loaded properly.
        Information is contained respectiely in attributes: V1n, V2n, V3n, V4n

        Examples:
          S1_.mat

        """
        info = DHP19Core.get_frame_info(path)
        x = np.swapaxes(io.loadmat(path)[f'V{info["cam"] + 1}n'], 0, 1)
        return x

    def get_frame_from_id(self, idx):
        return DHP19Core.load_frame(self.file_paths[idx])

    def get_label_from_id(self, idx):
        return self.classification_labels[idx]

    def get_joint_from_id(self, idx):
        joints_file = np.load(self.joints[idx])
        xyz = joints_file["xyz"].swapaxes(0, 1)
        intrinsic_matrix = torch.tensor(joints_file["camera"])
        extrinsic_matrix = torch.tensor(joints_file["M"])
        return Skeleton(xyz), intrinsic_matrix, extrinsic_matrix

    def get_heatmap_from_id(self, idx):
        hm_path = self.heatmaps[idx]
        return load_heatmap(hm_path, self.N_JOINTS)

    @staticmethod
    def _get_file_paths_with_cam_and_mov(data_dir, cams=None, movs=None):
        if cams is None:
            cams = [3]

        file_paths = np.array(get_file_paths(data_dir, extensions=[".npy", ".mat"]))
        cam_mask = np.zeros(len(file_paths))

        for c in cams:
            cam_mask += [f"cam_{c}" in x for x in file_paths]

        file_paths = file_paths[cam_mask > 0]
        if movs is not None:
            mov_mask = [
                DHP19Core.get_label_from_filename(x) in movs for x in file_paths
            ]

            file_paths = file_paths[mov_mask]

        return file_paths

    @staticmethod
    def get_frame_info(filename):
        filename = os.path.splitext(os.path.basename(filename))[0]

        result = {
            "subject": int(
                filename[filename.find("S") + 1 : filename.find("S") + 4].split("_")[0]
            ),
            "session": int(DHP19Core._get_info_from_string(filename, "session")),
            "mov": int(DHP19Core._get_info_from_string(filename, "mov")),
            "cam": int(DHP19Core._get_info_from_string(filename, "cam")),
            "frame": DHP19Core._get_info_from_string(filename, "frame"),
        }

        # # filename = os.path.splitext(os.path.basename(filename))[0]

        # result = {
        #     "subject": int(
        #         filename[filename.find("S") + 1 : filename.find("S") + 4].split("_")[0]
        #     ),
        #     "session": int(filename[filename.find('session') :].split('_')[0][7:]),
        #     "mov": int(filename[filename.find('mov') :].split('_')[0][3:]),
        #     # "cam": int(filename[filename.find('cam') :].split('_')[0]),
        #     # "frame": filename[filename.find('frame') :].split('_')[1],
        # }

        return result

    def get_test_subjects(self):
        return self.test_subjects

    def get_test_view(self):
        return self.test_cams


    @staticmethod
    def _get_info_from_string(filename, info, split_symbol="_"):
        return int(filename[filename.find(info) :].split(split_symbol)[1])

    @staticmethod
    def get_label_from_filename(filepath) -> int:
        """Given the filepath, return the correspondent movement label (range [0, 32])

        Args:
            filepath (str): frame absolute filepath

        Returns:
            Frame label

        Examples:
            >>> DHP19Core.get_label_from_filename("S1_session_2_mov_1_frame_249_cam_2.npy")
            8

        """

        label = 0
        info = DHP19Core.get_frame_info(filepath)

        for i in range(1, info["session"]):
            label += DHP19Core.MOVEMENTS_PER_SESSION[i]

        return label + info["mov"] - 1  # label in range [0, max_label)

    def _retrieve_data_files(self, labels_dir, suffix):
        labels_hm = [
            os.path.join(
                labels_dir, os.path.basename(x).split(".")[0] + suffix
            )
            for x in self.file_paths
        ]
        return labels_hm

    def train_partition_function(self, x):
        return self.frames_info[x]['subject'] not in self.test_subjects and self.frames_info[x]['cam'] not in self.test_cams



def load_heatmap(path, n_joints):
    joints = np.load(path)
    h, w = joints.shape
    y = np.zeros((h, w, n_joints))

    for joint_id in range(1, n_joints + 1):
        heatmap = (joints == joint_id).astype('float')
        if heatmap.sum() > 0:
            y[:, :, joint_id - 1] = decay_heatmap(heatmap)

    return y


def decay_heatmap(heatmap, sigma2=10):
    """

    Args
        heatmap :
           WxH matrix to decay
        sigma2 :
             (Default value = 1)

    Returns

        Heatmap obtained by gaussian-blurring the input
    """
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma2)
    heatmap /= np.max(heatmap)  # keep the max to 1
    return heatmap
