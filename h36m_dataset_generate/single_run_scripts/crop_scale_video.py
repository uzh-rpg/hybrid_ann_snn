import os
import re
import pdb
import subprocess

import numpy as np
from glob import glob
from pathlib import Path
import shlex


ffmpeg_binary = '/home/asude/ffmpeg-git-20220910-amd64-static/ffmpeg'
txt_filepath = '/home/asude/master_thesis/h36m_dataset_generate/joint_locs_per_sequence.txt'
with open(txt_filepath) as f:
    lines = f.readlines()

new_txt_filepath = '/home/asude/master_thesis/h36m_dataset_generate/crop_video_coordinates.txt'

root_path = '/longtermdatasets/asude/human3.6m_downloader/training/subject'
subjects = ['S1', 'S11',  'S5',  'S6',  'S7', 'S8',  'S9']
cam_id_map = {'54138969': 0, '55011271': 1, '58860488': 2, '60457274': 3}

# def runFFmpeg(commands):
#     if subprocess.run(commands).returncode == 0:
#         print ("FFmpeg Script Ran Successfully")
#     else:
#         print ("There was an error running your FFmpeg script")

def buildFFmpeg(action, action_dict):
    if action == 'crop':
        print('crop=960:768:{}:{}'.format(action_dict['x_start'], action_dict['y_start']))
        command = subprocess.run(shlex.split(ffmpeg_binary + ' -n -i {} -filter:v "crop=960:768:{}:{}" {}'.format(action_dict['video_name'], action_dict['x_start'], action_dict['y_start'], action_dict['crop_video_name'])))
    elif action == 'rescale':
        command = subprocess.run(shlex.split(ffmpeg_binary + ' -n -i {} -vf scale=320:256 {}'.format(action_dict['crop_video_name'], action_dict['rescale_video_name'])))
    else: 
        raise NotImplementedError
    return command

def build_dicts(video_dir, video_name, crop_xy):
    action_dict = {'video_name': video_dir + '/' + video_name + '.mp4', 
                    'crop_video_name': video_dir + '/' + 'crop_' + video_name + '.mp4',
                    'rescale_video_name': video_dir + '/' + 'rescale_' + video_name + '.mp4', 
                    'x_start': crop_xy[0], 
                    'y_start': crop_xy[1]}
    return action_dict

def get_subject_txt_file(lines, subject):
    subject = subject + ' '
    subject_lines = []
    subject_found = False
    for line in lines: 
        if subject in line: 
            subject_lines.append(line)
            if subject_found != True:
                subject_found = True
        else: 
            if subject_found: 
                return subject_lines

    return subject_lines

def read_video_name(video_name):
    vid_name_split = video_name.split('.')
    camera_val = vid_name_split[1]
    camera_id = cam_id_map[str(camera_val)]
    if '_' in vid_name_split[0]:
        tmp = vid_name_split[0].split('_')
        action_name = tmp[0]
        action_num = tmp[1]
    else: 
        action_name = vid_name_split[0]
        action_num = None
    return action_name, camera_id, action_num

def read_txt_file(video_name, subject_lines, action_name, camera_id, action_num):
    if action_num != None:
        search_name = '{} {} view {}'.format(action_name, action_num, camera_id)
    else:         
        search_name = '{} view {}'.format(action_name, camera_id)
    
    for line in subject_lines: 
        if search_name in line:
            return line
    
    return None

def get_xy_txt_file(video_info):
    xy_info = video_info.split('x_min')[-1].split(',')
    x_min = int(np.round(float(xy_info[0])))
    x_max = int(np.round(float(xy_info[1].split('x_max')[1])))
    y_min = int(np.round(float(xy_info[2].split('y_min')[1])))
    y_max = int(np.round(float(xy_info[3].split('y_max')[1])))
    return (x_min, x_max), (y_min, y_max)

def get_crop_dims(video_info, camera_id):
    xminmax, yminmax = get_xy_txt_file(video_info)

    image_height = 1000
    image_width = 1000

    if camera_id == 0 or camera_id == 3: 
        img_height = 1002
    
    human_height = 768 - (yminmax[1]-yminmax[0])
    adjust_crop_y = np.floor(human_height/2)
    y_crop = yminmax[0]-adjust_crop_y

    if y_crop < 0:
        y_crop = 0

    assert image_width > yminmax[0] >= 0
    assert image_height > yminmax[1] >= 0

    if xminmax[1] >= image_width:
        x_crop = 40
    else:
        human_width = 960 - (xminmax[1]-xminmax[0])
        adjust_crop_x = np.floor(human_width/2)
        x_crop = xminmax[0]-adjust_crop_x
    
    if x_crop < 0:
        x_crop = 0
    
    assert image_width > xminmax[0] >= 0
    assert image_width > x_crop >= 0
    assert image_height > y_crop >= 0

    return (int(x_crop), int(y_crop))

if __name__ == '__main__':
    with open(new_txt_filepath, 'w') as new_file:
        for subject in subjects: 

            subject_path = os.path.join(root_path, subject)
            assert Path(subject_path).exists()
            video_dirs = glob(subject_path + "/Videos/*", recursive = True)

            for video_dir in video_dirs:
                video_name = video_dir.split('/')[-1]

                subject_lines = get_subject_txt_file(lines, subject)
                video_name = video_name.replace("TakingPhoto", "Photo").replace("WalkingDog", "WalkDog")
                action_name, cam_id, action_num = read_video_name(video_name)
                video_info = read_txt_file(video_name, subject_lines, action_name, cam_id, action_num)
                print(video_name, video_info)
                assert video_info
                
                crop_xy = get_crop_dims(video_info, cam_id)
                action_dict = build_dicts(video_dir, video_name, crop_xy)

                if action_num != None:
                    new_file.write(subject + ' : ' + action_name + ' {}'.format(action_num) + ' view {}'.format(cam_id) + '   -->   ' + 'x_start {}, y_start {}'.format(crop_xy[0], crop_xy[1]) +' \n')
                else: 
                    new_file.write(subject + ' : ' + action_name + ' ' + ' view {}'.format(cam_id) + '   -->   ' + 'x_start {}, y_start {}'.format(crop_xy[0], crop_xy[1]) +' \n')

                action = 'crop'
                buildFFmpeg(action, action_dict)

                action = 'rescale'
                buildFFmpeg(action, action_dict)


