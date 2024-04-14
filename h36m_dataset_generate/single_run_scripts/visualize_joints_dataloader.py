'''
Train the CNN network with SNN compatible data. 
'''
import torch.nn as nn
import h5py
from configs.args import *

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam

from datetime import datetime
from torch_utils.network_initialization import init_weights, init_weights_div_last_layer
from models.cnn import UNet
from dataloaders.h36m_cnn_dataloader.provider import get_dataset
from torch_utils.visualization import *

if __name__ == '__main__':
    # config_path = '/home/asude/thesis_final/master_thesis/experiments/h36m_50hz_camview3/cnn_config_h36m_color.ini'
    config_path = '/home/asude/master_thesis/configs/cnn_config_h36m_color.ini'
    configfile = configparser.ConfigParser()
    configfile.read(config_path) # todo take from args

    # get config
    dataloader_config = configfile['dataloader_parameters'] 
    
    # SET DEVICE
    device = 'cpu'
    print('Running on', device)

    # PREPARE DATA LOADER
    dhp19_dataset = get_dataset(dataloader_config) 
    train_set = dhp19_dataset.get_train_dataset() 
    val_set = dhp19_dataset.get_test_dataset() 

    batch_size = 1
    params = {'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': True}

    train_loader = DataLoader(
        train_set,
        shuffle = True,
        **params
    )

    val_loader = DataLoader(
        val_set,
        shuffle = True,
        **params, 
    )

    def get_filename(filename):
        splits = filename[0].split('/')
        action_name = splits[-2]
        subject_name = splits[-4]
        return subject_name, action_name

    for batch_id, sample_batched in enumerate(val_loader):
        inputs = sample_batched['image'].to(device)
        labels = sample_batched['label'].to(device)
        gt_landmarks = sample_batched['landmarks'].to(device)
        # filename = sample_batched['filename']
        print(batch_id)
        # pdb.set_trace()
        # subject, action = get_filename(filename)
    
        plt.figure()
        plt.plot(gt_landmarks[0,0], gt_landmarks[0,1], 'r.')
        plt.imshow(torch.movedim(inputs[0], [0], [2]))
        # plt.title('{}: {}'.format(subject, action))
        plt.savefig('dataset_visualizations_rgb/img_{}.png'.format(batch_id))

    # Read a single image from original shape and overlay 2D pose
    # img_path = '/home/asude/thesis_final/master_thesis/h36m_dataset_generate/single_run_scripts/dataset_visualizations_mistake/Greeting.60457274/imgs/00000000.png'
    # label_path = '/media/mathias/moar/asude/h36m_events/labels.h5'

    # labels = h5py.File(label_path, 'r')
    # label = labels['S9']['Greeting']['cam_view_3'][0]
    # pdb.set_trace()
    # plt.figure()
    # img = plt.imread(img_path)
    # plt.imshow(img)
    # plt.scatter(np.round(label[0]), np.round(label[1]))
    # plt.savefig('img.png')