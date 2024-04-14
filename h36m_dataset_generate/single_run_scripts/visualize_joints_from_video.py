import pylab
import imageio
import pdb
import matplotlib.pyplot as plt
import numpy as np

'''
This script is used to read from mp4 videos, overlay 2D joints and test the dataset to ensure correct joint labels.
The subject id, action, and number of frames need to be configured.
'''
filename = '/home/asude/thesis_final/master_thesis/h36m_dataset_generate/single_run_scripts/dataset_visualizations_mistake/Waiting_1.60457274/Waiting_1.60457274.mp4'
# filename = '/home/asude/thesis_final/master_thesis/h36m_dataset_generate/single_run_scripts/dataset_visualizations_mistake/SittingDown_1.60457274/SittingDown_1.60457274.mp4'
# filename = '/home/asude/thesis_final/master_thesis/h36m_dataset_generate/single_run_scripts/dataset_visualizations_mistake/Greeting.60457274/Greeting.60457274.mp4'
# filename = '/longtermdatasets/asude/human3.6m_downloader/training/subject/S1/Videos/Directions_1.5413896/video/Directions_1.54138969.mp4'
label_filename = '/home/asude/thesis_final/master_thesis/h36m_dataset_generate/single_run_scripts/dataset_visualizations_mistake/data_2d_h36m_gt.npz'
label_npz = np.load(label_filename, allow_pickle=True)
label_2d = label_npz['positions_2d'].tolist()
label_2d_vid = label_2d['S9']['Waiting 1'][3]

vid = imageio.get_reader(filename,  'ffmpeg')
nums = np.arange(0,1400,60)
for num in nums:
    image = vid.get_data(num)
    fig = plt.figure()
    fig.suptitle('image #{}'.format(num), fontsize=20)
    plt.plot(label_2d_vid[num,:,0], label_2d_vid[num,:,1], 'r.')
    plt.plot(0, 0, 'g.')
    plt.imshow(image)
    plt.savefig('dataset_visualizations_mistake/label_test_original_all_labels{}.png'.format(num))
    pdb.set_trace()