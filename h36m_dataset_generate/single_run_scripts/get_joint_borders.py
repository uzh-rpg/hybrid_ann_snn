import numpy as np
import pdb 

label_filename = '/longtermdatasets/asude/human3.6m_downloader/training/data_2d_h36m_gt_13joints.npz'

lbl_handle = np.load(label_filename, allow_pickle=True)
labels = lbl_handle['positions_2d'].tolist()

subjects = ['S1', 'S11',  'S5',  'S6',  'S7', 'S8',  'S9']
cam_views = [0,1,2,3]

# largest_width, largest_height
largest_subject_bounding_box = np.zeros(2)
smallest_image_view = [1e6, 0, 1e6, 0]

pdb.set_trace()
# largest_subject_bounding_box_dict={ '0': np.zeros(2),
#                                     '1': np.zeros(2), 
#                                     '2': np.zeros(2),
#                                     '3': np.zeros(2),
#                                     }

# x_min, x_max, y_min, y_max
# stab = [1e6, 0, 1e6, 0]
# smallest_image_view_dict={  '0': np.zeros(4) + stab,
#                             '1': np.zeros(4) + stab, 
#                             '2': np.zeros(4) + stab,
#                             '3': np.zeros(4) + stab,
#                             }


# largest_width = 0
# largest_height = 0

# x_max = 0
# x_min = 1e6
# y_max = 0
# y_min = 1e6

def do_comparison(x_loc, y_loc, largest_subject_bounding_box, smallest_image_view):

    largest_width, largest_height = largest_subject_bounding_box[0], largest_subject_bounding_box[1]

    x_min = smallest_image_view[0]
    x_max = smallest_image_view[1]
    y_min = smallest_image_view[2]
    y_max = smallest_image_view[3]

    x_loc_max = x_loc.max() 
    x_loc_min = x_loc.min() 
    y_loc_max = y_loc.max() 
    y_loc_min = y_loc.min() 

    width = x_loc_max - x_loc_min
    if width > largest_width:
        largest_width = width

    height = y_loc_max - y_loc_min
    if height > largest_height:
        largest_height = height

    if x_loc_max > x_max:
        x_max = x_loc_max

    if x_loc_min < x_min:
        x_min = x_loc_min

    if y_loc_max > y_max:
        y_max = y_loc_max

    if y_loc_min < y_min:
        y_min = y_loc_min

    return [largest_width, largest_height], [x_min, x_max, y_min, y_max]

def get_vals_per_seq(x_loc, y_loc):

    x_max = x_loc.max() 
    x_min = x_loc.min() 
    y_max = y_loc.max() 
    y_min = y_loc.min() 

    largest_width = x_max - x_min

    largest_height = y_max - y_min

    return [largest_width, largest_height], [x_min, x_max, y_min, y_max]

x_max_val = 0
y_max_val = 0 
with open('joint_locs_per_sequence.txt', 'w') as f:
    for subject in subjects: 
        for action in labels[subject].keys():
            for cam_view in cam_views:
                # for frame_num in range(len(labels[subject][action][cam_view])):
                x_loc = labels[subject][action][cam_view][:,:,0]
                y_loc = labels[subject][action][cam_view][:,:,1]
                
                x_y_diff, x_y_edge = get_vals_per_seq(x_loc, y_loc)
                f.write(subject + ' : ' + action + ' view {}'.format(cam_view) + '   -->   ' + 'width {}, height {}'.format(x_y_diff[0], x_y_diff[1]) +  ' x_min {}, x_max {}, y_min {}, y_max {}'.format(x_y_edge[0], x_y_edge[1], x_y_edge[2], x_y_edge[3]) +' \n')
                
                if x_y_diff[0] > x_max_val:
                    x_max_val = x_y_diff[0]
                if x_y_diff[1] > y_max_val:
                    y_max_val = x_y_diff[1]
                # pdb.set_trace()
print(x_max_val, y_max_val)
# print('Largest subject bounding box has width {} and height {}'.format(largest_width, largest_height))
# print('Smallest view for x = ({}, {}), for y = ({}, {})'.format(x_min, x_max, y_min, y_max))

# print('Largest subject bounding box dict', largest_subject_bounding_box)
# print('Smallest view dict', smallest_image_view)

                # f.write(subject + '  : ' + action + '   -->  ' + str(frame_num) + '  {}, {}'.format(x_loc.min(), x_loc.max()) + '  {}, {}'.format(y_loc.min(), y_loc.max())+ ' \n')
                    # pdb.set_trace()

