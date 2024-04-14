import copy
import os
import re
import pdb
from typing import List

import numpy as np
import torch
from kornia import quaternion_to_rotation_matrix
from pose3d_utils.camera import CameraIntrinsics

from experimenting.utils import Skeleton

from ..utils import get_file_paths
from .base import BaseCore
from .h3m import h36m_cameras_extrinsic_params, h36m_cameras_intrinsic_params


class HumanCore(BaseCore):
    """
    Human3.6m core class. It provides implementation to load frames, 2djoints, 3d joints

    """

    RECORDING_HZ = 50
    CAMS_ID_MAP = {'54138969': 0, '55011271': 1, '58860488': 2, '60457274': 3}
    JOINTS = [15, 25, 17, 26, 18, 1, 6, 27, 19, 2, 7, 3, 8]
    LABELS_MAP = {
        'Directions': 0,
        'Discussion': 1,
        'Eating': 2,
        'Greeting': 3,
        'Phoning': 4,
        'Posing': 5,
        'Purchase': 6,
        'Sitting': 7,
        'SittingDown': 8,
        'Smoking': 9,
        'Photo': 10,
        'Waiting': 11,
        'Walking': 12,
        'WalkDog': 13,
        'WalkTogether': 14,
        'Purchases': 15,
        '_ALL': 15,
    }
    MAX_WIDTH = 260  # DVS resolution
    MAX_HEIGHT = 346  # DVS resolution
    N_JOINTS = 13
    N_CLASSES = 2
    TORSO_LENGTH = 453
    DEFAULT_TEST_SUBJECTS: List[int] = [9, 11]
    DEFAULT_TEST_VIEW = [4]
    DEFAULT_TRAIN_VIEW = [0, 1, 2, 3]

    def __init__(
        self,
        name,
        data_dir,
        joints_path,
        partition,
        n_channels,
        movs=None,
        test_subjects=None,
        train_cams=None,
        test_cams=None,
        avg_torso_length=TORSO_LENGTH,
        *args,
        **kwargs,
    ):
        super(HumanCore, self).__init__(name, partition)

        self.file_paths = HumanCore._get_file_paths_with_movs(data_dir, movs)

        self.in_shape = (HumanCore.MAX_HEIGHT, HumanCore.MAX_WIDTH)
        self.n_channels = n_channels

        self.avg_torso_length = avg_torso_length

        self.classification_labels = [HumanCore.get_label_from_filename(x_path) for x_path in self.file_paths]

        self.n_joints = HumanCore.N_JOINTS
        self.joints = HumanCore.get_pose_data(joints_path)
        
        self.frames_info = [HumanCore.get_frame_info(x) for x in self.file_paths]

        self.timestamps_mask = self.get_timestamps_mask()
        if test_subjects is None:
            self.test_subjects = HumanCore.DEFAULT_TEST_SUBJECTS
        else:
            self.test_subjects = test_subjects

        if train_cams is None:
            self.train_cams = HumanCore.DEFAULT_TRAIN_VIEW
        else:
            self.train_cams = train_cams

        if test_cams is None:
            self.view = HumanCore.DEFAULT_TEST_VIEW
        else:
            self.view = test_cams

    def get_test_subjects(self):
        return self.test_subjects

    def get_test_view(self):
        return self.view

    @staticmethod
    def get_label_from_filename(filepath) -> int:
        """
        Given the filepath, return the correspondent movement label (range [0, 32])

        Args:
            filepath (str): frame absolute filepath

        Returns:
            Frame label

        Examples:
            >>> HumanCore.get_label_from_filename("S1_session_2_mov_1_frame_249_cam_2.npy")
        """

        info = HumanCore.get_frame_info(filepath)

        return HumanCore.LABELS_MAP[info['action'].split(" ")[0]]

    
    def get_frame_info_from_id(self, x):
        return self.frames_info[x]

    @staticmethod
    def get_frame_info(filepath):
        """
        >>> HumanCore.get_label_frame_info("tests/data/h3m/S1/Directions 1.54138969S1/frame0000001.npy")
        {'subject': 1, 'actions': 'Directions', cam': 0, 'frame': '0000007'}
        """
        base_subject_dir = filepath[re.search(r'(?<=S)\d+/', filepath).span()[1] :]
        infos = base_subject_dir.split('/')

        cam = re.search(r"(?<=\.)\d+", base_subject_dir)
        cam = HumanCore.CAMS_ID_MAP[cam.group(0)] if cam is not None else None
        # result = {
        #     "subject": int(re.search(r'(?<=S)\d+', filepath).group(0)),
        #     "action": re.findall(r"(\w+\s?\d?)\.\d+", base_subject_dir)[0],
        #     "cam": cam,
        #     "frame": re.search(r"\d+", infos[-1]).group(0),
        # }
        action_name = re.findall(r"(\w+\s?\d?)\.\d+", base_subject_dir)[0]
        action_name = action_name.replace('_', ' ')
        
        result = {
            "subject": int(re.search(r'(?<=S)\d+', filepath).group(0)),
            "action": action_name,
            "cam": cam,
            "frame": re.search(r"\d+", infos[-1]).group(0),
        }

        return result

    @staticmethod
    def _get_file_paths_with_movs(data_dir, movs):
        file_paths = np.array(get_file_paths(data_dir, ['npy']))
        if movs is not None:
            mov_mask = [
                HumanCore.get_label_from_filename(x) in movs for x in file_paths
            ]

            file_paths = file_paths[mov_mask]
        return file_paths

    def get_timestamps_mask(self) -> np.ndarray:
        """
        Return indexes corresponding to multiple of 64th frame of each recording
        """
        data_indexes = np.arange(len(self.file_paths))
        mask_every_n_frame = 64

        freq = mask_every_n_frame / self.RECORDING_HZ
        last = 0
        mask = np.full(len(self.file_paths), False)

        for i in data_indexes:
            try:
                t = self.try_get_timestamp_from_id(i)
                
                if t < last:
                    last = 0
                if t - last > freq:
                    mask[i] = True
                    last = t
            except:
                continue
        
        return mask

    @staticmethod
    def get_pose_data(path: str) -> dict:
        """
        Parse npz file and extract gt information (3d joints, timestamps, ...)
        """
        data = np.load(path, allow_pickle=True)
        result = {}

        result = HumanCore._get_joints_data(data, result)
        result = HumanCore._get_timestamps_data(data, result)
        return result

    @staticmethod
    def _get_timestamps_data(data, result: dict):
        result = copy.deepcopy(result)
        
        if 'timestamps' not in data:
            return result

        data = data['timestamps'].item()

        for subject, actions in data.items():
            subject_n = int(re.search(r"\d+", subject).group(0))
            if subject_n not in result:
                result[subject_n] = {}

            for action_name, timestamps in actions.items():
                if action_name not in result[subject_n]:
                    result[subject_n][action_name] = {}
                result[subject_n][action_name]['timestamps'] = timestamps

        return result

    @staticmethod
    def _get_joints_data(data, result):
        result = copy.deepcopy(result)

        data = data['positions_3d'].item()

        for subject, actions in data.items():
            subject_n = int(re.search(r"\d+", subject).group(0))
            if subject_n not in result:
                result[subject_n] = {}

            for action_name, positions in actions.items():
                if action_name not in result[subject_n]:
                    result[subject_n][action_name] = {}
                    result[subject_n][action_name]['positions'] = positions
                    result[subject_n][action_name][
                        'extrinsics'
                    ] = h36m_cameras_extrinsic_params[subject]

        return result

    @staticmethod
    def _build_intrinsic(intrinsic_matrix_params: dict) -> torch.Tensor:
        # scale to DVS frame dimension
        w_ratio = HumanCore.MAX_WIDTH / intrinsic_matrix_params['res_w']
        h_ratio = HumanCore.MAX_HEIGHT / intrinsic_matrix_params['res_h']

        intr_linear_matrix = torch.tensor(
            [
                [
                    w_ratio * intrinsic_matrix_params['focal_length'][0],
                    0,
                    w_ratio * intrinsic_matrix_params['center'][0],
                    0,
                ],
                [
                    0,
                    h_ratio * intrinsic_matrix_params['focal_length'][1],
                    h_ratio * intrinsic_matrix_params['center'][1],
                    0,
                ],
                [0, 0, 1, 0],
            ]
        )
        return intr_linear_matrix

    @staticmethod
    def _build_extrinsic(extrinsic_matrix_params: dict) -> torch.Tensor:

        quaternion = torch.tensor(extrinsic_matrix_params['orientation'])[[1, 2, 3, 0]]
        quaternion[:3] *= -1

        R = quaternion_to_rotation_matrix(quaternion)
        t = torch.tensor(extrinsic_matrix_params['translation'])
        tr = -torch.matmul(R, t)

        return torch.cat([torch.cat([R, tr.unsqueeze(1)], dim=1)], dim=0,)

    def get_joint_from_id(self, idx):
        joints_data = self._get_joint_from_id(idx)

        intr_matrix, extr_matrix = self.get_matrices_from_id(idx)
        return Skeleton(joints_data), intr_matrix, extr_matrix

    def get_matrices_from_id(self, idx):
        frame_info = self.get_frame_info_from_id(idx)
        intr_matrix = HumanCore._build_intrinsic(
            h36m_cameras_intrinsic_params[frame_info['cam']]
        )

        extr = self.joints[frame_info['subject']][frame_info['action']]['extrinsics'][
            frame_info['cam']
        ]
        extr_matrix = HumanCore._build_extrinsic(extr)
        return intr_matrix, extr_matrix

    def _get_id_from_path(self, path):
        return np.where(self.file_paths == path)

    def _get_joint_from_id(self, idx):
        frame_info = self.get_frame_info_from_id(idx)
        frame_n = int(frame_info['frame'])
        joints_data = self.joints[frame_info['subject']][frame_info['action']][
            'positions'
        ][frame_n]
        joints_data = joints_data[HumanCore.JOINTS] * 1000  # Scale to cm
        return joints_data

    def try_get_timestamp_from_id(self, idx):
        try:
            frame_info = self.get_frame_info_from_id(idx)
            frame_n = int(frame_info['frame'])
            timestamp = self.joints[frame_info['subject']][frame_info['action']][
                'timestamps'
            ][frame_n]
        except:
            print("Timestamps missing")
            raise Exception("Timestamp not found")

        return timestamp

    def get_frame_from_id(self, idx):
        path = self.file_paths[idx]
        x = np.load(path, allow_pickle=True) / 255.0
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
        return x

    def get_cross_subject_partition_function(self):
        """
        Get partition function for cross-subject evaluation method. Keep 64th frame

        Note:
          Core class must implement get_test_subjects
          get_frame_info must provide frame's subject
        """

        return lambda x: (
            self.frames_info[x]['subject'] in self.get_test_subjects()
            and self.timestamps_mask[x]
        )

    def get_cross_view_partition_function(self):
        """
        Get partition function for cross-subject evaluation method. Keep 64th frame

        Note:
          Core class must implement get_test_subjects
          get_frame_info must provide frame's subject
        """

        return lambda x: (
            self.frames_info[x]['subject'] in self.get_test_subjects()
            and self.timestamps_mask[x]
            and self.frames_info[x]['cam'] in self.get_test_view()
        )

    def train_partition_function(self, x):
        return self.frames_info[x]['subject'] not in self.test_subjects and self.frames_info[x]['cam'] in self.train_cams
