"""
Author: Gianluca Scarpellini - gianluca.scarpellini@iit.it
"""

from typing import Tuple
import pdb
import numpy as np


def normalized_3sigma(input_img: np.ndarray) -> np.ndarray:
    img = input_img.copy().astype('float')

    sig_img = img[img > 0].std()
    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255
    numSdevs = 3.0
    range = numSdevs * sig_img

    img[img != 0] *= 255 / range
    img[img < 0] = 0
    img[img > 255] = 255

    return img.astype('uint8')


def voxel_grid_joint_generator(
    events: np.ndarray,
    joints: np.ndarray,
    num_events: int,
    frame_size: Tuple[int, int],
    n_cameras: int = 4,
    n_bins: int = 4,
) -> np.ndarray:
    """
    Generate constant_count frames and corresponding gt 3D joints labels. 3D joints labels were acquired at 200fps
    """
    voxel_frame = np.zeros(
        (n_cameras, frame_size[0], frame_size[1], n_bins), dtype="int"
    )

    upper_bound = len(joints) * 1 / 200
    init_slice = 0
    t0 = events[0][2]
    dt = events[num_events][2] - events[0][2]

    for ind, event in enumerate(events):
        y = int(event[0])
        x = int(event[1])
        ti = event[2]
        pi = event[3]

        cam = int(event[-1])  # using camera info similar to DHP19

        voxel_frame[cam, x, y] += 1

        t_split = (n_bins - 1) / dt * (ti - t0) + 1
        for t_bin in range(0, n_bins):
            voxel_frame[cam, x, y, t_bin] += pi * max(0, 1 - np.abs(t_bin - t_split))

        if ti > upper_bound:
            # Recording ends here
            return

        if (ind + 1) % num_events == 0:

            for idx in range(n_cameras):
                voxel_frame[idx] = normalized_3sigma(voxel_frame[idx])

            yield voxel_frame, ti
            voxel_frame = np.zeros_like(voxel_frame)

            init_slice = ind
            final_slice = min(init_slice + num_events - 1, len(events) - 1)
            t0 = events[init_slice, 2]
            dt = events[final_slice, 2] - t0


def joint_generator(events: np.ndarray, joints: np.ndarray, num_events: int) -> np.ndarray: # TODO: fix seems incorrect
    """
    Generate constant_count frames and corresponding gt 3D joints labels. 3D joints labels were acquired at 200fps
    """
    start_joint_data_index = 0
    joint_data_fps = 200
    upper_bound = len(joints) * 20 / 1000

    for ind, event in enumerate(events):
        ti = event[2]

        if ti > upper_bound:
            # Recording ends here
            return

        if (ind + 1) % num_events == 0:
            end_joint_data_index = int(ti * 1e3/20) + 1 # index of end label for duration
            joints_per_frame = np.nanmean(
                joints[start_joint_data_index:end_joint_data_index, :], 0 # get mean of all labels during that duration
            )

            yield joints_per_frame

            start_joint_data_index = end_joint_data_index


def constant_count_joint_generator( # TODO: fix seems incorrect
    events: np.ndarray,
    joints: np.ndarray,
    num_events: int,
    frame_size: Tuple[int, int],
    n_cameras: int = 4,
) -> np.ndarray:
    """
    Generate constant_count frames and corresponding gt 3D joints labels. 3D joints labels were acquired at 200fps
    """
    event_count_frame = np.zeros((n_cameras, frame_size[0], frame_size[1]), dtype="int")

    upper_bound = len(joints) * 20 / 1000 # why is this here?

    for ind, event in enumerate(events): # gets a single event
        y = int(event[0])
        x = int(event[1])
        t = event[2]
        cam = int(event[-1])  # using camera info similar to DHP19

        event_count_frame[cam, x, y] += 1

        if t > upper_bound:
            # Recording ends here
            return

        if (ind + 1) % num_events == 0: # when total number of events reaches num_events=30000, normalize and save
            for idx in range(n_cameras):
                event_count_frame[idx] = normalized_3sigma(event_count_frame[idx]) # normalize all cam_views seperately 

            yield event_count_frame, t
            event_count_frame = np.zeros_like(event_count_frame)


def timestamps_generator(
    events: np.ndarray,
    joints: np.ndarray,
    num_events: int,
    n_cameras: int = 4,
) -> np.ndarray:
    """
    Generate constant_count frames and corresponding gt 3D joints labels. 3D joints labels were acquired at 200fps
    """
    upper_bound = len(joints) * 1 / 200

    for ind, event in enumerate(events):
        t = event[2]
        if t > upper_bound:
            # Recording ends here
            return

        if (ind + 1) % num_events == 0:

            yield None, t
