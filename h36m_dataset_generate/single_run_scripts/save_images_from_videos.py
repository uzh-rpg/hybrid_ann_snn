import cv2
import skvideo.io
import os
import pdb
import numpy as np

from pathlib import Path
from fractions import Fraction

def write_to_img(img: np.ndarray, idx: int, imgs_dir: str):

    assert os.path.isdir(imgs_dir)
    # img = np.clip(img * 255, 0, 255).astype("uint8")
    path = os.path.join(imgs_dir, "%08d.png" % idx)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


root_path = '/longtermdatasets/asude/human3.6m_downloader/training/subject/'
for root, dirs, files in os.walk(root_path): 
    for file_name in files: 
        if file_name[:7] == 'rescale': 
            img_path = os.path.join(root, 'imgs') 
            
            if not Path(img_path).exists():
                os.mkdir(img_path) # 1 prepare outdir

            video_path = os.path.join(root, file_name) 
            metadata = skvideo.io.ffprobe(os.path.abspath(video_path))
            
            fps = float(Fraction(metadata['video']['@avg_frame_rate']))
            fps = fps
            num_frames = int(metadata['video']['@nb_frames'])

            videogen = skvideo.io.vreader(os.path.abspath(video_path)) # 2 read video 
        
            for idx, frame in enumerate(videogen):
                write_to_img(frame, idx, img_path) # 3 write to img 