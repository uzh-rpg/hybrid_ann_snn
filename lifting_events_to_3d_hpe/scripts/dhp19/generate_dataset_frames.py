"""
Tool for converting DHP19 frames generated with matlab to numpy, and with standard path format
"""

import argparse
import glob
import os
import sys
import pdb 
from pathlib import Path
import h5py
import numpy as np

from experimenting.dataset.core import DHP19Core

sys.path.insert(0, "../")


def convert_raw_frame_and_save(x_path, out_dir):
    filename = os.path.basename(x_path)
    sub = filename[filename.find('S') + 1 : filename.find('S') + 4].split('_')[0]
    session = int(filename[filename.find('session') + len('session')])
    mov = int(filename[filename.find('mov') + len('mov')])

    x_h5 = h5py.File(x_path, 'r')
    for cam in range(4):
        frames = x_h5['DVS'][..., cam]

        for ind in list(range(len(frames))):
            frame = frames[ind, :, :]
            frame_path = os.path.join(
                out_dir, DHP19Core.get_standard_path(sub, session, mov, ind, cam, "")
            )
            out_path = os.path.join(out_dir, frame_path.format(ind, cam))
            if Path(out_path).exists():
                continue
            else:
                np.save(out_path, frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert DHP19 standard frames to our format'
    )

    parser.add_argument(
        '--input_dir',
        nargs='+',
        help='Input dir path of frames generated using matlab scripts',
    )
    parser.add_argument('--out_dir', nargs='+', help='Output dir path')
    args = parser.parse_args()

    root_dir = args.input_dir[0]
    out_dir = args.out_dir[0]

    x_paths = sorted(glob.glob(os.path.join(root_dir, "*events.h5")))
    n_files = len(x_paths)

    for x_path in x_paths:
        convert_raw_frame_and_save(x_path, out_dir)
