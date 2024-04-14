import argparse
import glob
import os
from os.path import join
import pdb
import h5py
import numpy as np
from tqdm import tqdm

from experimenting import utils
from experimenting.dataset.core import DHP19Core

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GT from DHP19')

    parser.add_argument('--input_dir', type=str, help='Input path generated from DHP19')
    parser.add_argument('--out_dir', type=str, help='Output dir')
    parser.add_argument('--p_matrices_dir', type=str, help='P_matrices DHP19')
    args = parser.parse_args()

    out_dir = args.out_dir
    input_dir = args.input_dir

    p_mat_dir = args.p_matrices_dir
    os.makedirs(out_dir, exist_ok=True)
    width = 344
    height = 260

    p_mat_cam1 = np.load(join(p_mat_dir, 'P1.npy'))
    p_mat_cam2 = np.load(join(p_mat_dir, 'P2.npy'))
    p_mat_cam3 = np.load(join(p_mat_dir, 'P3.npy'))
    p_mat_cam4 = np.load(join(p_mat_dir, 'P4.npy'))
    p_mats = [p_mat_cam4, p_mat_cam1, p_mat_cam3, p_mat_cam2]

    x_paths = sorted(glob.glob(join(input_dir, "*label.h5")))

    n_files = len(x_paths)
    print(f"N of files: {n_files}")

    for x_path in tqdm(x_paths):

        filename = os.path.basename(x_path)
        info = DHP19Core.get_frame_info(filename)

        sub = info['subject']
        session = info['session']
        mov = info['mov']
        out_label_path = os.path.join(out_dir, "S{}_session_{}_mov_{}_frame_".format(sub, session, mov) + "{}_cam_{}_2dhm",)

        x_h5 = h5py.File(x_path, 'r')

        frames = x_h5['XYZ']  # JOINTS xyz
        for cam in range(4):  # iterating cams (0, 1, 2, 3)
            extrinsics_matrix, camera_matrix = utils.decompose_projection_matrix(p_mats[cam])

            for ind in list(range(len(frames))):
                xyz = frames[ind, :]
                out_filename = out_label_path.format(ind, cam)
                out_path = os.path.join(out_dir, out_filename)

                np.savez(
                    out_path,
                    xyz=xyz,
                    M=extrinsics_matrix,
                    camera=camera_matrix,
                )
