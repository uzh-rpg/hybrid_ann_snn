import numpy as np 
import pdb
'''
This script was used to select the 13 joints used in the Event Human3.6m dataset from the 32 joints that were provided in the original Human3.6m dataset. 
The 3D labels are saved and converted to 2D labels using Video Pose from Facebook.
'''
def get_subset_joints(input_path):
    JOINTS = [15, 25, 17, 26, 18, 1, 6, 27, 19, 2, 7, 3, 8]
    all_joints = np.load(input_path, allow_pickle=True)
    all_joints_list = all_joints['positions_3d'].tolist()
    for subject_name in all_joints_list.keys():
        for action_name in all_joints_list[subject_name].keys():
            all_joints_list[subject_name][action_name] = all_joints_list[subject_name][action_name][:,JOINTS,:]
    return all_joints_list
    # pdb.set_trace()

def save_3d_label_to_npz(output_dir, joints):
    np.savez(output_dir, positions_3d=joints)

input_path = '/longtermdatasets/asude/human3.6m_downloader/training/data_3d_h36m.npz'
selected_joints = get_subset_joints(input_path)

output_dir = '/longtermdatasets/asude/human3.6m_downloader/training/data_3d_h36m_13joints.npz'
save_3d_label_to_npz(output_dir, selected_joints)