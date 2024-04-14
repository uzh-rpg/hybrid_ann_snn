# Generate DHP19
We provide two methods to generate frames from DHP19 dataset. Both requires to
download ground truth, DAVIS data, and cameras projection matrix from
[download](https://sites.google.com/view/dhp19/download?authuser=0). We call
this directory `rootDataFolder`.

## Generate frames
### Method 1 
1. Use DHP19 tools (you can find them at https://github.com/SensorsINI/DHP19).
   This generates `**events.h5` and `*labels.h5*` files in `outDatasetFolder`
2. Use python script `generate_dataset_frames.py --input_dir outDatasetFolder
   --out_dir output_frames_dir` to separate frames for different cameras. You should get:
   ```bash
   +-- output_frames_dir 
   | ...
   | +-- S1_session_5_mov_7_frame_86_cam_0.npy 
   | +-- S1_session_5_mov_7_frame_86_cam_1.npy 
   | +-- S1_session_5_mov_7_frame_86_cam_2.npy
   | +-- S1_session_5_mov_7_frame_86_cam_3.npy
   | ...
   ```
3. To generate the labels, check the following `generate labels`.

### Method 2
It's provided a toolset for generating `.mat` event frames from `DHP19` dataset.
The supported representation are: `spatiotemporal voxelgrid`, `time surfaces`
and `constant count`. In `Generate_DHP.m`, fix `rootCodeFolder`,
`rootDataFolder` and `outDatasetFolder` to your setup. 

You must modify `Generate_DHP19` according to your need:
a. You can generate `constant-count` frames along with labels by setting the
   extract function to `ExtractEventsToFramesAndMeanLabels`
b. You can generate other representations with `ExtractEventsToVoxel` or
   `ExtractEventsToTimeSurface`

After setting the extract function in the script, launch
```
matlab -r "run('./scripts/dhp19/generate_DHP19/Generate_DHP19.m')"
```

To generate the labels, check the following `generate labels`.

## Generate labels
Use python script `generate_joints.py --input_dir outDatasetFolder --out_dir
output_labels_dir --p_matrices_dir rootDataFolder/P_matrices` to generate npz
files from DHP19 labels. The output tree path should be:

```bash
   +-- output_labels_dir 
   | ...
   | +-- S1_session_1_mov_1_frame_86_cam_0_2dhm.npz
   | +-- S1_session_1_mov_1_frame_86_cam_1_2dhm.npz
   | +-- S1_session_1_mov_1_frame_86_cam_2_2dhm.npz
   | +-- S1_session_1_mov_1_frame_86_cam_3_2dhm.npz
   | ...
   ```

## Help
You can ask for help either contacting me at `gianluca.scarpellini[at]iit.it` or opening an issue!
