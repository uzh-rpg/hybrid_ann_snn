# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# We added some minor adjustments. You can find the original file at https://github.com/facebookresearch/VideoPose3D

import argparse
import os
import sys
import zipfile
from glob import glob
from shutil import rmtree

import h5py
import numpy as np

sys.path.append("../")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Converts .mat joint files to .npz"
    )
    parser.add_argument("--input_dir", help="file(s) to convert to output")

    args = parser.parse_args()
    return args


output_filename = "data_3d_h36m"
subjects = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

if __name__ == "__main__":
    args = parse_args()

    # Convert dataset from original source, using files converted to .mat (the Human3.6M dataset path must be specified manually)
    # This option requires MATLAB to convert files using the provided script
    if os.path.exists(output_filename + ".npz"):
        print("The dataset already exists at", output_filename + ".npz")
        exit(0)

    if args.input_dir:
        print("Converting original Human3.6M dataset from", args.input_dir)
        output = {}

        from scipy.io import loadmat

        for subject in subjects:
            output[subject] = {}
            file_list = glob(
                # Full instead of D3_positions
                args.input_dir
                + "/"
                + subject
                + "/MyPoseFeatures/FULL_D3_Positions/*.mat"
            )
            assert len(file_list) == 30, (
                "Expected 30 files for subject "
                + subject
                + ", got "
                + str(len(file_list))
                + ". Have you generated FULL_D3_Positions? Check the readme"
            )

            for f in file_list:
                action = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]

                if subject == "S11" and action == "Directions":
                    continue  # Discard corrupted video

                # Use consistent naming convention
                canonical_name = action.replace("TakingPhoto", "Photo").replace(
                    "WalkingDog", "WalkDog"
                )

                hf = loadmat(f)
                if "F" in hf:
                    data = hf["F"]
                elif "data" in hf:
                    data = hf["data"]
                positions = data[0, 0].reshape(-1, 32, 3)
                positions /= 1000  # Meters instead of millimeters
                output[subject][canonical_name] = positions.astype("float32")

        print("Saving...")
        np.savez_compressed(output_filename, positions_3d=output)

        print("Done.")
    else:
        print(
            "Specify valid input dir. This should be the base_dir of your h3m dataset"
        )
