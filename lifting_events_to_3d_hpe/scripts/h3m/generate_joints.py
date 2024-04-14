import argparse
import os
import pdb
import pathlib
import re 

import event_library as el
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from utils import (
    constant_count_joint_generator,
    joint_generator,
    timestamps_generator,
    voxel_grid_joint_generator,
)

from experimenting.dataset.core.h3mcore import HumanCore

def get_frame_info_(filepath, base_dir):
    """
    >>> HumanCore.get_label_frame_info("tests/data/h3m/S1/Directions 1.54138969S1/frame0000001.npy")
    {'subject': 1, 'actions': 'Directions', cam': 0, 'frame': '0000007'}
    """
    action = filepath.replace('_', ' ')
    subject = base_dir.split('/')[-2]

    result = {
        "subject": subject,
        "action": action, 
    }

    return result

def load_from_numpy(file_path: str, num_events: int = -1) -> np.ndarray:
    events = np.load(file_path, allow_pickle=True)
    # normalize events time starting at 0
    events[:, 2] -= events[0, 2]
    return events

def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_unique_action_paths(file_path: str) -> np.ndarray:
    subdirectories = get_subdirectories(file_path)

    event_files = []
    for d in subdirectories: 
        filename = d.split('.')[0]
        if filename not in event_files: 
            event_files.append(filename)
    return event_files

def parse_args():
    parser = argparse.ArgumentParser(
        description="Accumulates events to an event-frame."
    )
    parser.add_argument("--event_files", nargs="+", help="file(s) to convert to output")
    parser.add_argument(
        "--joints_file",
        type=str,
        help="file of .npz joints containing joints data. Generate it using `prepare_data_h3m`",
    )
    parser.add_argument("--output_base_dir", type=str, help="output_dir")
    parser.add_argument(
        "--representation",
        type=str,
        default='constant-count',
        help="representation to use for generating events frames. Choose between [constant-count, voxel-grid]",
    )
    parser.add_argument(
        '--generate-joints',
        action='store_true',
        help='If set, generate also joints labels synchronized with event frames',
    )
    parser.add_argument(
        "--num_events", type=int, default=30000, help="num events to accumulate"
    )

    args = parser.parse_args()
    return args


def _get_multicam_events(event_file, n_cameras=4):
    'event files should contain 4 files with the different 4 camera views of the same action in order cam_view [0,1,2,3]'
    events = []
    for camera_idx in range(0, n_cameras): 
        cam_code = cam_index_to_id_map[camera_idx]
        event_file_dir = os.path.join(event_files_dir, event_file + '.{}'.format(cam_code))
        assert Path(event_file_dir).exists()
        filename = os.path.join(event_file_dir, 'frame0000000.npy')
        events.append(load_from_numpy(filename)) # (4, (2606635, 4)) (num_views, (num_events, xytp))

    events_tmp = [np.concatenate([events[index], index * np.ones((len(events[index]), 1))], axis=1) for index in range(n_cameras)]

    events_concat = np.concatenate(events_tmp)
    sort_index = np.argsort(events_concat[:, 2])
    final_events = events_concat[sort_index] # sort events wrt timestamp
    return final_events


# def _generate_timestamps(events: np.array, joints: np.array) -> np.array:

#     ts_generator = timestamps_generator(events, joints, num_events)
#     timestamps = []
#     for ts_frame in ts_generator:
#         timestamps.append(ts_frame)
#     return np.stack(timestamps)

def _generate_joints(events: np.array, joints: np.array) -> np.array:

    gt_generator = joint_generator(events, joints, num_events)
    joints = []
    for joint_frame in gt_generator:
        joints.append(joint_frame)
    return np.stack(joints)


if __name__ == '__main__':
    args = parse_args()
    joints_file = args.joints_file
    root_dir = args.event_files
    num_events = args.num_events
    data = HumanCore.get_pose_data(joints_file)
    hw_info = el.utils.get_hw_property('dvs')
    output_base_dir = args.output_base_dir

    n_cameras = 4  # Number of parallel cameras
    switch = {
        'constant-count': constant_count_joint_generator,
        'voxel-grid': voxel_grid_joint_generator,
    }
    # os.makedirs(output_base_dir)
    output_joint_path = os.path.join(output_base_dir, "3d_joints")

    joint_gt = {f"S{s:01d}": {} for s in range(1, 12)} # 11 subjects
    timestamps = {f"S{s:01d}": {} for s in range(1, 12)} # 11 subjects

    cam_index_to_id_map = dict(zip(HumanCore.CAMS_ID_MAP.values(), HumanCore.CAMS_ID_MAP.keys()))

    representation_generator = switch[args.representation] # constant count or voxel-grid generator
    # representation_generator = timestamps_generator

    subjects = ['S9', 'S11']
    for subject in subjects:
        event_files_dir = os.path.join(root_dir[0], subject, 'Events')

        def _fun(event_file):
            info = get_frame_info_(event_file, event_files_dir)
            action = info['action'].replace(' ', '_')
            subject = info['subject']
            action = action.replace("TakingPhoto", "Photo").replace("WalkingDog", "WalkDog")

            if subject == 'S11' and action == "Directions":
                print(f"Discard {info}")
                return

            if "_ALL" in action:
                print(f"Discard {info}")
                return

            output_dir = os.path.join(output_base_dir, f"{subject}", f"{action}")

            timestamps[subject][action] = []
            joints = data[int(re.search(r'\d+', subject).group())][action.replace('_',' ')]['positions']

            events = _get_multicam_events(event_file, n_cameras)
            frame_generator = representation_generator(events, joints, num_events, hw_info.size) 

            for ind_frame, events_per_cam in enumerate(frame_generator): # what is frame generator 
                event_frame_per_cams, timestamp = events_per_cam # (4, 346, 260) 4 cam views and timestamp
                # print(ind_frame, timestamp)
                for id_camera in range(n_cameras): # iterate over 4 camera views
                    cam_code = cam_index_to_id_map[id_camera] # camera code correspoding to camera id
                    out_const_count_dir = output_dir.replace('_', ' ') + '.{}'.format(cam_code)
                    os.makedirs(out_const_count_dir, exist_ok=True) # create file and save constant count representation
                    np.save(os.path.join(out_const_count_dir, f"frame{ind_frame:07d}.npy"), event_frame_per_cams[id_camera])
            if args.generate_joints:
                joint_gt[subject][action.replace('_',' ')] = _generate_joints(events, joints)
                # timestamps_generator[f"S{info['subject']:01d}"][action] = _generate_timestamps(events, timestamps)

        event_files = get_unique_action_paths(event_files_dir)
        for event_file in event_files:
            _fun(event_file)

    # thread_map(_fun, list(range(0, len(event_files), n_cameras)), max_workers=16)
    # save_params = {"timestamps": timestamps} # unused
    save_params = {}
    if args.generate_joints:
        save_params["positions_3d"] = joint_gt

    np.savez_compressed(output_joint_path, **save_params)
