import pdb

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import io
import sys
sys.path.append("..") # Adds higher directory to python modules path.
# import matplotlib
# matplotlib.use('Agg')

from PIL import Image
from torchvision.utils import make_grid
from matplotlib.lines import Line2D
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import gaussian_blur
from torch_utils.utils import *
import torchvision.transforms.functional as tf

def plot_labeled_batch(sample_batched):
    '''
    Show image with gt labels for a batch of samples.
    Note: labels are gaussian blured to make learning easier. (Visualization includes these blurs.)
    '''

    if 'frame' in sample_batched:
        images_batch = sample_batched['frame']
    else: # for histogram event representation in image space
        images_batch = torch.sum(sample_batched['event_representation'], dim=1)
        images_batch = torch.unsqueeze(images_batch, dim=1)

    labels_batch = gaussian_blur_label(sample_batched['label'].float())
    grid_images = make_grid(images_batch, nrow=4)
    grid_labels = make_grid(labels_batch, nrow=4)
    plt.imshow(grid_labels.numpy().transpose(1, 2, 0).sum(axis=2))
    plt.imshow(grid_images.numpy().transpose(1, 2, 0), cmap='gray', alpha=0.2)
    plt.axis('off')
    plt.title('Dataloader batch: label and frame overlaid')

def plot_sample_from_batch(sample_batched, sample_id):
    '''
    Show single image from a batch: frame and label overlaid.
    '''
    plt.figure()
    images_sample = sample_batched['frame'][sample_id]
    labels_sample = gaussian_blur_label(sample_batched['label'][sample_id].float())
    plt.imshow(labels_sample.sum(axis=0))
    plt.imshow(images_sample[0], cmap='gray', alpha=0.2)
    plt.axis('off')
    plt.title('Batch from dataloader')

def plot_pred_heatmaps(landmarks, output, points = 'all'):
    '''
    Plot heatmaps of the 13 channels - 1 for each joint.
    Overlaid with gt landmarks as an indication of where the blob should be located.
    Overlaid with pred landmarks (max pixel value in each channel)
    Currently implemented for only batch size 1
    '''

    pred_landmarks = get_pred_landmarks(output)

    num_img_row = output.shape[0]
    num_img_col = output.shape[1]

    grid_padding = 0
    output_grid = torch.squeeze(output, dim=0)
    output_grid = torch.unsqueeze(output_grid, dim=1)
    grid_images = make_grid(output_grid, nrow = num_img_col, ncol = num_img_row, padding = grid_padding)

    grid_border_size = grid_padding

    img_height, img_width = output.size(2), output.size(3)
    # img_scale = 1.5
    # plt.figure(figsize=(img_scale*6.4, img_scale*4.8), dpi=img_scale*100)
    plt.imshow(grid_images.detach().numpy().transpose((1, 2, 0)))
    plt.gcf().set_size_inches(60, 40)

    if points == 'all':
        num_bins = landmarks.shape[2]
        num_joints = landmarks.shape[3]
        weights = np.arange(0, num_bins, 1)
        cmap_val = 'YlGn'

        for i in range(num_img_col):
            for j in range(num_joints):
                if num_img_row == 1:
                    plt.scatter(landmarks[:, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                landmarks[:, 1, :, j].numpy(), s=10, marker='.', c=weights, cmap=cmap_val)

                else:
                    plt.scatter(landmarks[:, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                landmarks[:, 1, :, j].numpy() + int(i/num_img_col) * img_height + (int(i/num_img_col) + 1) * grid_border_size,
                                s=10, marker='.', c=weights, cmap=cmap_val)

            plt.scatter(pred_landmarks[:, 0, :] + int(i % num_img_col) * img_width + (int(i % num_img_col) + 1) * grid_border_size,
                        pred_landmarks[:, 1, :] + int((i / num_img_col)) * img_height + int((i / num_img_col) + 1) * grid_border_size, s=10, marker='.', c='b')

        plt.colorbar(shrink=0.1)
    else:
        # mark joint locations of prediction and groundtruth
        for i in range(num_img_col):
            if num_img_row == 1:
                plt.scatter(landmarks[:, 0, -1].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            landmarks[:, 1, -1].numpy(), s=10, marker='.', c='r')

                plt.scatter(pred_landmarks[:, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            pred_landmarks[:, 1, :] + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size, s=10, marker='.', c='b')
            else:
                plt.scatter(landmarks[:, 0, -1].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            landmarks[:, 1, -1].numpy() + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                            s=10, marker='.', c='r')

                plt.scatter(pred_landmarks[:, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            pred_landmarks[:, 1, :] + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                            s=10, marker='.', c='b')

        plt.legend(['gt', 'pred'], ncol = 2, loc = 'lower right', fontsize = 'small')

    plt.axis('off')
    image = plot_to_tensor()
    return image

def plot_batch_label_pred_tb(sample_batched, output, points = 'all', grid_shape = None, plot_init=False):
    '''
    Plot frames, ground truth joint locations, prediction joint locations by extracting location of maximum value pixels
    using makegrid for saving in tensorboard. Grid size: (batch_size/4, 4)
    '''
    if 'image' in sample_batched:
        images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    else:
        hist_batch, landmarks_batch = sample_batched['event_representation'], sample_batched['landmarks']
        batch = torch.sum(hist_batch, dim=1)
        if batch.dim() == 4:
            if batch.shape[1] == 1: # if channel size is 1
                batch = torch.squeeze(batch, dim=1)
            elif batch.shape[1] == 2:
                batch = torch.sum(batch, dim=1)
            else:
                raise NotImplementedError
        batch_norm = norm_image_tensor(batch)
        images_batch = torch.unsqueeze(batch_norm, dim=1)

    # if pred landmarks is provided for all time steps (10), instead of the output
    if output.dim()==4 and output.shape[2] == 10:
        pred_landmarks = output
    else:
        pred_landmarks = get_pred_landmarks(output)

    batch_size = len(images_batch)
    if grid_shape == None:
        num_img_col = 4
        assert not batch_size%4
        num_img_row = int(batch_size/num_img_col)
    else:
        num_img_row = grid_shape[0]
        num_img_col = grid_shape[1]

    grid_padding = 0
    grid_images = make_grid(images_batch, nrow = num_img_col, padding = grid_padding)
    grid_border_size = grid_padding

    img_height, img_width = images_batch.size(2), images_batch.size(3)
    img_scale = 1.5
    plt.figure(figsize=(img_scale*6.4, img_scale*4.8), dpi=img_scale*100)
    plt.imshow(grid_images.numpy().transpose((1, 2, 0)), cmap='gray', alpha=0.5)
    # plt.gcf().set_size_inches(60, 40)

    # no time axis, channel dimension is summed 
    if landmarks_batch.dim() == 3:
        for i in range(batch_size):
            if num_img_row == 1:
                plt.scatter(landmarks_batch[i, 0, :].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            landmarks_batch[i, 1, :].numpy(), s=10, marker='.', c='r', label='gt')

                plt.scatter(pred_landmarks[i, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            pred_landmarks[i, 1, :] + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size, s=10, marker='.', c='g', label='gt')
            else:
                plt.scatter(landmarks_batch[i, 0, :].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            landmarks_batch[i, 1, :].numpy() + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                            s=10, marker='.', c='r', label='pred')

                plt.scatter(pred_landmarks[i, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            pred_landmarks[i, 1, :] + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                            s=10, marker='.', c='g', label='pred')

        # plt.legend(ncol = 2, loc = 'lower right', fontsize = 'small')

    else: # 4 dimensions including time axis, channel dimension is summed. For this label init plotting is possible.
        if 'label_init' in sample_batched and plot_init == True:
            init_landmarks = get_pred_landmarks(sample_batched['label_init'].to('cpu'))
            
            if num_img_row == 1:
                for i in range(batch_size):
                    plt.scatter(init_landmarks[i, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                        init_landmarks[i, 1, :], s=10, marker='.', c='indigo', label='init')
            else:
                for i in range(batch_size):
                    plt.scatter(init_landmarks[i, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                        init_landmarks[i, 1, :] + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                        s=10, marker='.', c='indigo', label='init')
                    
        # plots all labels in sequence 
        if points == 'all':
            num_bins = landmarks_batch.shape[2]
            num_joints = landmarks_batch.shape[3]
            weights = np.arange(0, num_bins, 1)
            cmap_val = 'cool'
            for i in range(batch_size): # iterates over samples in batch 
                for j in range(num_joints): # iterates over 13 joints in each sample 
                    if num_img_row == 1:
                        plt.scatter(landmarks_batch[i, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                    landmarks_batch[i, 1, :, j].numpy(), s=10, marker='.', c=weights, cmap=cmap_val)

                    else:
                        plt.scatter(landmarks_batch[i, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                    landmarks_batch[i, 1, :, j].numpy() + int(i/num_img_col) * img_height + (int(i/num_img_col) + 1) * grid_border_size,
                                    s=10, marker='.', c=weights, cmap=cmap_val)
                
                if pred_landmarks.dim() == 4: # if all predictions in a sequence should be plotted. 
                # Need to pass in all predictions from a sequence when calling the plot function.
                    for j in range(num_joints):
                        if num_img_row == 1:
                            plt.scatter(pred_landmarks[i, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                        pred_landmarks[i, 1, :, j].numpy(), s=10, marker='.', c=weights, cmap='winter')

                        else:
                            plt.scatter(pred_landmarks[i, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                        pred_landmarks[i, 1, :, j].numpy() + int(i/num_img_col) * img_height + (int(i/num_img_col) + 1) * grid_border_size,
                                        s=10, marker='.', c=weights, cmap='winter')
                
                else: # plot only last label prediction (or equivalently the prediction that was passed in)
                    plt.scatter(pred_landmarks[i, 0, :] + int(i % num_img_col) * img_width + (int(i % num_img_col) + 1) * grid_border_size,
                                pred_landmarks[i, 1, :] + int((i / num_img_col)) * img_height + int((i / num_img_col) + 1) * grid_border_size, s=10, marker='.', c='g')

            # plt.colorbar(shrink=0.5)
        else: # for plotting only the last label and its prediction in a sequence 
            for i in range(batch_size):
                if num_img_row == 1:
                    plt.scatter(landmarks_batch[i, 0, -1].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                landmarks_batch[i, 1, -1].numpy(), s=10, marker='.', c='r', label='gt')

                    plt.scatter(pred_landmarks[i, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                pred_landmarks[i, 1, :] + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size, s=10, marker='.', c='g', label='pred')
                else:
                    plt.scatter(landmarks_batch[i, 0, -1].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                landmarks_batch[i, 1, -1].numpy() + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                                s=10, marker='.', c='r', label='gt')

                    plt.scatter(pred_landmarks[i, 0, :] + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                pred_landmarks[i, 1, :] + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                                s=10, marker='.', c='g', label='pred')
            # if plot_init:
            #      plt.legend(ncol = 3, loc = 'lower right', fontsize = 'small')
            # else:
            #     plt.legend(ncol = 2, loc = 'lower right', fontsize = 'small')

    plt.axis('off')
    image = plot_to_tensor()
    return image

def plot_batch_label_pred_tb_h36m(sample_batched, output, mode, img_color=None, grid_shape=None, cnn_preds=None):
    '''
    Plot frames, ground truth joint locations, prediction joint locations by extracting location of maximum value pixels
    using makegrid for saving in tensorboard. Grid size: (batch_size/4, 4)
    '''
    if mode == 'cnn':
        batch_size = 8
        if img_color == 'RGB':
            plot_rgb_image_tb(sample_batched, batch_size, grid_shape)
        elif img_color == 'grayscale':
            plot_gray_image_tb(sample_batched, batch_size, grid_shape)
        else: 
            plot_constcount_tb(sample_batched, batch_size, grid_shape)
        plot_cnn_labels_tb(sample_batched, output, batch_size, grid_shape)
    elif mode == 'snn':
        batch_size = 2
        plot_events_tb(sample_batched, batch_size, (1,2))
        plot_snn_labels_tb(sample_batched, output, batch_size, (1, 2))
    elif mode == 'hybrid':
        batch_size = sample_batched['image'].shape[0]
        if sample_batched['image'].shape[1] == 1: # grayscale image
            plot_gray_image_tb(sample_batched, batch_size, grid_shape, transparency=0.4)
        elif sample_batched['image'].shape[1] == 3: # rgb image
            plot_rgb_image_tb(sample_batched, batch_size, grid_shape, transparency=0.4)
        elif sample_batched['image'].shape[1] > 3: # constant count
            plot_constcount_tb(sample_batched, batch_size, grid_shape, transparency=0.4)
        plot_events_tb(sample_batched, batch_size, grid_shape, cmap='gray_r')
        if cnn_preds != None:
            plot_hybrid_cnn_labels_tb(sample_batched, cnn_preds, batch_size, grid_shape)
        plot_snn_labels_tb(sample_batched, output, batch_size, grid_shape)
    else: 
        NotImplementedError

    plt.axis('off')
    image = plot_to_tensor()
    return image

def plot_constcount_tb(sample_batched, batch_size, grid_shape, transparency=0.5):
    images_batch = sample_batched['image'][:batch_size]
    images_batch = images_batch.sum(dim = 1)
    
    batch_norm = norm_image_tensor(images_batch)
    batch_norm = torch.unsqueeze(batch_norm, dim=1)

    num_img_row = grid_shape[0]
    num_img_col = grid_shape[1]

    grid_images = make_grid(batch_norm, nrow = num_img_col, padding = 0)
    img_height, img_width = batch_norm.size(2), batch_norm.size(3)
    
    img_scale = 1.5
    plt.figure(figsize=(img_scale*6.4, img_scale*4.8), dpi=img_scale*100)
    plt.imshow(grid_images.numpy().transpose((1, 2, 0)), alpha=transparency)


def plot_rgb_image_tb(sample_batched, batch_size, grid_shape, transparency=0.5):
    images_batch = sample_batched['image'][:batch_size] # take first 8 samples
    
    num_img_row = grid_shape[0]
    num_img_col = grid_shape[1]

    grid_images = make_grid(images_batch, nrow = num_img_col, padding = 0)
    img_height, img_width = images_batch.size(2), images_batch.size(3)
    
    img_scale = 1.5
    plt.figure(figsize=(img_scale*6.4, img_scale*4.8), dpi=img_scale*100)
    plt.imshow(grid_images.numpy().transpose((1, 2, 0)), alpha=transparency)

def plot_gray_image_tb(sample_batched, batch_size, grid_shape, transparency=0.5):
    images_batch = sample_batched['image'][:batch_size] # take first 8 samples
    
    num_img_row = grid_shape[0]
    num_img_col = grid_shape[1]

    grid_images = make_grid(images_batch, nrow = num_img_col, padding = 0)
    img_height, img_width = images_batch.size(2), images_batch.size(3)
    
    img_scale = 1.5
    plt.figure(figsize=(img_scale*6.4, img_scale*4.8), dpi=img_scale*100)
    plt.imshow(grid_images.numpy().transpose((1, 2, 0)), cmap='gray', alpha=transparency)

def plot_hybrid_cnn_labels_tb(sample_batched, output, batch_size, grid_shape):

    pred_landmarks = get_pred_landmarks(output).cpu().numpy()
    
    img_height, img_width = sample_batched['image'].size(2), sample_batched['image'].size(3)

    num_img_row = grid_shape[0]
    num_img_col = grid_shape[1]

    for i in range(batch_size): # iterates over samples in batch 
        plt.scatter(pred_landmarks[i, 0, :] + int(i % num_img_col) * img_width, pred_landmarks[i, 1, :] + int(i / num_img_col) * img_height, s=6, marker='.', color='indigo')

def plot_cnn_labels_tb(sample_batched, output, batch_size, grid_shape):

    pred_landmarks = get_pred_landmarks(output[:batch_size])
    landmarks_batch = sample_batched['landmarks'][:batch_size]
    
    img_height, img_width = sample_batched['image'].size(2), sample_batched['image'].size(3)

    assert batch_size >= 8
    num_img_row = grid_shape[0]
    num_img_col = grid_shape[1]

    for i in range(batch_size): # iterates over samples in batch 
        plt.scatter(landmarks_batch[i, 0, :].numpy() + int(i%num_img_col) * img_width, landmarks_batch[i, 1, :].numpy() + int(i / num_img_col) * img_height, s=6, marker='.', color='indigo')
        plt.scatter(pred_landmarks[i, 0, :] + int(i % num_img_col) * img_width, pred_landmarks[i, 1, :] + int(i / num_img_col) * img_height, s=6, marker='.', color='deepskyblue')

def plot_events_tb(sample_batched, batch_size, grid_shape, cmap='gray'):
    images_batch = sample_batched['event_representation'] 
    img_height, img_width = images_batch.size(3), images_batch.size(4)
    images_batch = images_batch.sum(dim=[1,2])
    batch_norm = norm_image_tensor(images_batch)
    batch_norm = torch.unsqueeze(batch_norm, dim=1)
    
    num_img_row = grid_shape[0]
    num_img_col = grid_shape[1]

    grid_images = make_grid(batch_norm, nrow = num_img_col, padding = 0)

    img_scale = 1.5
    if cmap == 'gray':
        plt.figure(figsize=(img_scale*6.4, img_scale*4.8), dpi=img_scale*100)
    plt.imshow(grid_images.numpy().transpose((1, 2, 0)), cmap=cmap, alpha=0.5)

def plot_snn_labels_tb(sample_batched, output, batch_size, grid_shape):

    pred_landmarks = get_pred_landmarks(output)
    gt_landmarks = sample_batched['landmarks']
    
    img_height, img_width = sample_batched['event_representation'].size(3), sample_batched['event_representation'].size(4)

    num_img_row = grid_shape[0]
    num_img_col = grid_shape[1]

    for i in range(batch_size): # iterates over samples in batch 
        for k in range(13):
            plt.scatter(gt_landmarks[i,:,0,k].numpy() + int(i%num_img_col) * img_width, gt_landmarks[i,:,1,k].numpy() + int(i / num_img_col) * img_height, s=10, marker='.', c=np.arange(0,gt_landmarks.shape[1],1), cmap='cool')
        plt.scatter(pred_landmarks[i,0] + int(i % num_img_col) * img_width, pred_landmarks[i,1] + int(i / num_img_col) * img_height, s=10, marker='.', color='g')


def plot_batch_label_tb(sample_batched, points = 'all', grid_shape = None):
    '''
    Plot frames and corresponding ground truth joint locations
    using makegrid for saving in tensorboard. If grid_shape None: then grid size => (batch_size/4, 4)
    '''
    if 'frame' in sample_batched:
        images_batch, landmarks_batch = sample_batched['frame'], sample_batched['landmarks']
    else:
        hist_batch, landmarks_batch = sample_batched['event_representation'], sample_batched['landmarks']
        batch = torch.sum(hist_batch, dim=1)
        if batch.shape[1] == 1:
            batch = torch.squeeze(batch, dim=1)
        elif batch.shape[1] == 2:
            batch = torch.sum(batch, dim=1)
        else:
            raise NotImplementedError
        batch_norm = norm_image_tensor(batch)
        images_batch = torch.unsqueeze(batch_norm, dim=1)

    batch_size = len(images_batch)
    if grid_shape == None:
        num_img_col = 4
        assert not batch_size%4
        num_img_row = int(batch_size/num_img_col)
    else:
        num_img_row = grid_shape[0]
        num_img_col = grid_shape[1]

    grid_padding = 0
    grid_images = make_grid(images_batch, nrow = num_img_col, padding = grid_padding)
    grid_border_size = grid_padding

    img_height, img_width = images_batch.size(2), images_batch.size(3)
    img_scale = 1.5
    plt.figure(figsize=(img_scale*6.4, img_scale*4.8), dpi=img_scale*100)
    plt.imshow(grid_images.numpy().transpose((1, 2, 0)), cmap='gray', alpha=0.5)

    if points == 'all':
        num_bins = landmarks_batch.shape[2]
        num_joints = landmarks_batch.shape[3]
        weights = np.arange(0, num_bins, 1)
        cmap_val = 'cool'
        print('grid size:',num_img_row, num_img_col)
        for i in range(batch_size):
            for j in range(num_joints):
                if num_img_row == 1:
                    plt.scatter(landmarks_batch[i, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                landmarks_batch[i, 1, :, j].numpy(), s=10, marker='.', c=weights, cmap=cmap_val)
                else:
                    plt.scatter(landmarks_batch[i, 0, :, j].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                                landmarks_batch[i, 1, :, j].numpy() + int(i/num_img_col) * img_height + (int(i/num_img_col) + 1) * grid_border_size,
                                s=10, marker='.', c=weights, cmap=cmap_val)

        plt.colorbar(shrink=0.5)
    else:
        weights = 'r'
        cmap_val = None
        for i in range(batch_size):
           if num_img_row == 1:
                plt.scatter(landmarks_batch[i, 0, -1].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                            landmarks_batch[i, 1, -1].numpy(),
                            s=10, marker='.', c=weights, cmap = cmap_val)
           else:
               plt.scatter(landmarks_batch[i, 0, -1].numpy() + int(i%num_img_col) * img_width + (int(i%num_img_col) + 1) * grid_border_size,
                           landmarks_batch[i, 1, -1].numpy() + int((i/num_img_col)) * img_height + int((i/num_img_col) + 1)  * grid_border_size,
                           s=10, marker='.', c=weights, cmap = cmap_val)

    plt.axis('off')
    plt.legend(['gt'], ncol = 1, loc = 'lower right', fontsize = 'small')
    image = plot_to_tensor()
    return image

def plot_to_tensor():
    '''
    Convert matplotlib plots to tensor for compatability with tensorboard
    '''
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)

    buf.seek(0)
    # open image from buffer with PIL
    pil_image = Image.open(buf)
    # PIL to tensor
    tensor_image = ToTensor()(pil_image)
    return tensor_image

def plot_grad_flow(named_parameters, lr=1):
    '''
    Retrieved from Daniel Gehrig on 21/02/22.

    From:
    https://gist.github.com/danielgehrig18/f149c300017111790f5bdf6ef30e796b
    '''
    
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    figsize = (25,10)
    fig, ax = plt.subplots(figsize=figsize)

    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and p.grad is not None:
            layers.append(n)
            ave_grads.append(lr*p.grad.cpu().abs().mean())
            max_grads.append(lr*p.grad.cpu().abs().max())
            min_grads.append(lr*p.grad.cpu().abs().min())

    ax.bar(3*np.arange(len(max_grads)), max_grads, lw=2, color="r")
    ax.bar(3*np.arange(len(max_grads)), ave_grads, lw=2, color="m")
    ax.bar(3*np.arange(len(max_grads)), min_grads, lw=2, color="b")

    ax.set_xticks(range(0, 3*len(ave_grads), 3))
    labels = ax.set_xticklabels(layers)
    for l in labels:
        l.update({"rotation": "vertical"})

    ax.set_xlim(left=0, right=3*len(ave_grads))
    ax.set_ylim(bottom=1e-7*lr, top=lr*1e-1) 
    ax.set_yscale("log")# zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="r", lw=4),
               Line2D([0], [0], color="m", lw=4),
               Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient', 'min-gradient'])

    fig.tight_layout()
    image = plot_to_tensor()

    return image

def plot_weight_distribution(named_parameters):
    '''
    Adapted from Daniel Gehrig's plot_grad_flow from Daniel Gehrig on 21/02/22.

    From:
    https://gist.github.com/danielgehrig18/f149c300017111790f5bdf6ef30e796b
    '''
    
    '''Plots the weights in different layers in the net for weight visualization.
    call "plot_weight_distribution(self.model.named_parameters())" to visualize the weights'''
    figsize = (20,10)
    fig, ax = plt.subplots(figsize=figsize)

    ave_weights = []
    max_weights = []
    min_weights = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad):
            layers.append(n)
            layer_weights = p.cpu().detach()
            ave_weights.append(layer_weights.abs().mean())
            max_weights.append(layer_weights.abs().max())
            min_weights.append(layer_weights.abs().min())

    ax.bar(3*np.arange(len(max_weights)), max_weights, lw=2, color="r")
    ax.bar(3*np.arange(len(max_weights)), ave_weights, lw=2, color="m")
    ax.bar(3*np.arange(len(max_weights)), min_weights, lw=2, color="b")

    ax.set_xticks(range(0, 3*len(ave_weights), 3))
    labels = ax.set_xticklabels(layers)
    for l in labels:
        l.update({"rotation": "vertical"})

    ax.set_xlim(left=0, right=3*len(ave_weights))
    ax.set_yscale("log")# zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average weight")
    ax.set_title("Weights")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="r", lw=4),
               Line2D([0], [0], color="m", lw=4),
               Line2D([0], [0], color="b", lw=4)], ['max-weight', 'mean-weight', 'min-weight'])

    fig.tight_layout()
    image = plot_to_tensor()

    return image

def plot_tau_distribution(named_parameters):
    '''
    Adapted from Daniel Gehrig's plot_grad_flow from Daniel Gehrig on 21/02/22.

    From:
    https://gist.github.com/danielgehrig18/f149c300017111790f5bdf6ef30e796b
    '''
    
    '''Plots the time constant in different layers in the network.
    call "plot_tau_distribution(self.model.named_parameters())" to visualize the time constant values'''
    figsize = (10,10)
    fig, ax = plt.subplots(figsize=figsize)

    ave_tau = []
    max_tau = []
    min_tau = []
    layers = []
    for name, param in named_parameters:
        if name[-2:] == '.w':
            layers.append(name)
            layer_tau = (1./param.sigmoid()).cpu().detach()
            ave_tau.append(layer_tau.mean())
            max_tau.append(layer_tau.max())
            min_tau.append(layer_tau.min())

    ax.bar(3*np.arange(len(max_tau)), max_tau, lw=2, color="r")
    ax.bar(3*np.arange(len(max_tau)), ave_tau, lw=2, color="m")
    ax.bar(3*np.arange(len(max_tau)), min_tau, lw=2, color="b")

    ax.set_xticks(range(0, 3*len(ave_tau), 3))
    labels = ax.set_xticklabels(layers)
    for l in labels:
        l.update({"rotation": "vertical"})

    ax.set_xlim(left=0, right=3*len(ave_tau))
    ax.set_yscale("linear")# zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average tau")
    ax.set_title("Membrane Time Constants")
    ax.grid(True)
    ax.legend([Line2D([0], [0], color="r", lw=4),
               Line2D([0], [0], color="m", lw=4),
               Line2D([0], [0], color="b", lw=4)], ['max-tau', 'mean-tau', 'min-tau'])

    fig.tight_layout()
    image = plot_to_tensor()

    return image

def generate_video(ims, sample):
    '''
    Generate video from single torch samples and labels. Currently only 2 image overlay is possible.
    Initialize empty figure before starting iterations. eg. fig = plt.figure()

    args:
        video_path: directory of video to save
        ims: list to be filled with each iteration - needs to be initialized with an empty list
        sample: dataset dictionary containing 'frame' and 'label' for a single sample

        ims: list of images to append new image in every iteration - needs to be passed in
    '''

    frame_pil = tf.to_pil_image(sample['frame'][0])
    label = sample['label'].float()
    blur_label = gaussian_blur_label(label)
    labels_pil = tf.to_pil_image(torch.sum(blur_label, axis=0))
    overlay_img = Image.blend(frame_pil, labels_pil, 0.8)

    im = plt.imshow(overlay_img, animated=True)
    ims.append([im])
    plt.axis('off')

    return ims

# def save_imgs_ev_dataset(sample, frame_id, save_dir):

def generate_video_w_pred(ims, sample, pred_landmarks, title, ax):
    '''
    Generate video from single torch samples, labels, and their predictions.
    Initialize empty figure before starting iterations. eg. fig, ax = plt.subplots()

    args:
        video_path: directory of video to save
        ims: list to be filled with each iteration - needs to be initialized with an empty list
        sample: dataset dictionary containing 'frame' and 'label' for a single sample
        title: a different title can be set for each frame

        ims: list of images to append new image in every iteration - needs to be passed in
    '''

    num_bins = 10
    num_joints = 13

    if 'frame' in sample:
        frame = sample['frame'][0].numpy()
    else:
        hist_batch = sample['event_representation']
        hist_sum = torch.sum(hist_batch[0], dim=0) # sum time bins
        if hist_sum.shape[0] == 2:
            hist_sum = torch.sum(hist_sum, dim=0)
        frame = norm_image(hist_sum.numpy())
    # pdb.set_trace()
    gt_landmarks = sample['landmarks']
    label_init = sample['label_init']
    init_landmarks = get_pred_landmarks(label_init)
    # pred_landmarks = pred_landmarks[0].numpy()

    im = ax.imshow(frame, cmap = 'gray', alpha=0.5, animated=True)

    weights = np.arange(0, num_bins, 1)

    j = 0
    sct_landmarks_0 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j], 
    s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 1
    sct_landmarks_1 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 2
    sct_landmarks_2 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 3
    sct_landmarks_3 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j], 
    s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 4
    sct_landmarks_4 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 5
    sct_landmarks_5 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 6
    sct_landmarks_6 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j], 
    s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 7
    sct_landmarks_7 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 8
    sct_landmarks_8 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 9
    sct_landmarks_9 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j], 
    s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 10
    sct_landmarks_10 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 11
    sct_landmarks_11 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)
    j = 12
    sct_landmarks_12 = ax.scatter(gt_landmarks[0, 0, :, j], gt_landmarks[0, 1, :, j],
            s=10, marker='.', c=weights, cmap='cool', animated=True)

    sct_init = ax.scatter(init_landmarks[0, 0, :], init_landmarks[0, 1, :],
                s=10, marker='.', c='g', animated=True)
    # sct_pred = ax.scatter(pred_landmarks[0, 0, 0], pred_landmarks[0, 1, 0],
    #             s=10, marker='.', c='b', animated = True)
    # plt.colorbar(shrink=0.5)
    
    if title == None: 
        title = 'Initialization t = {} ms, Event window: [{}, {}] ms'.format(int(sample['timestamp'].item()/1e3), sample['ms_start'].item(), sample['ms_end'].item())


    ttl = plt.text(0.5, 1.01, title, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
    ims.append([im, sct_landmarks_0, sct_landmarks_1, sct_landmarks_2, sct_landmarks_3, sct_landmarks_4, sct_landmarks_5, \
    sct_landmarks_6, sct_landmarks_7, sct_landmarks_8, sct_landmarks_9, sct_landmarks_10, sct_landmarks_11, \
    sct_landmarks_12, sct_init, ttl])
    # ims.append([im, sct_gt, ttl])

    plt.axis('off')
    # plt.legend(['gt', 'pred'], ncol = 2, loc = 'lower right', fontsize = 'small')
    # plt.legend(['gt'], ncol = 1, loc = 'lower right', fontsize = 'small')

    return ims

def generate_video_w_pred_bin_by_bin(ims, sample, min_landmarks, pred_landmarks, title, ax, bin_id, init_flag=True):
    '''
    Generate video from single torch samples, labels, and their predictions.
    Initialize empty figure before starting iterations. eg. fig, ax = plt.subplots()

    args:
        video_path: directory of video to save
        ims: list to be filled with each iteration - needs to be initialized with an empty list
        sample: dataset dictionary containing 'frame' and 'label' for a single sample
        title: a different title can be set for each frame

        ims: list of images to append new image in every iteration - needs to be passed in
    '''

    num_bins = 10
    num_joints = 13

    if 'frame' in sample:
        frame = sample['frame'][0].numpy()
    else:
        # pdb.set_trace()
        hist_batch = sample['event_representation'][0, bin_id] # select 1st batch
        if hist_batch.shape[0] == 2:
            hist_sum = torch.sum(hist_batch, dim=0) # if channel size is 2
        assert hist_sum.dim() == 2
        frame = norm_image(hist_sum.numpy())

    gt_landmarks = sample['landmarks']
    init_landmarks = sample['label_init']
    # init_landmarks = get_pred_landmarks(label_init)

    assert pred_landmarks.dim() == 3
    assert init_landmarks.dim() == 3
    assert gt_landmarks.dim() == 4
    pred_landmarks = pred_landmarks.numpy()

    im = ax.imshow(frame, cmap = 'gray', alpha=0.5, animated=True)

    # pdb.set_trace()

    if init_flag:
        sct_init = ax.scatter(init_landmarks[0, 0, :], init_landmarks[0, 1, :],
        s=10, marker='.', c='g', animated=True)

    sct_landmarks = ax.scatter(gt_landmarks[0, 0, bin_id, :], gt_landmarks[0, 1, bin_id, :], 
    s=20, marker='.', c='b', animated=True)

    if min_landmarks != None:
        min_pred = ax.scatter(min_landmarks[0, 0, :], min_landmarks[0, 1, :], 
        s=10, marker='.', c='orange', animated=True)
    
    max_pred = ax.scatter(pred_landmarks[0, 0, :], pred_landmarks[0, 1, :], 
    s=20, marker='.', c='r', animated=True)

    ttl = plt.text(0.5, 1.01, title, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
    if min_landmarks != None:
        if init_flag:
            ims.append([im, sct_init, sct_landmarks, min_pred, max_pred, ttl])
        else:
            ims.append([im, sct_landmarks, min_pred, max_pred, ttl])

    else: 
        if init_flag:
            ims.append([im, sct_init, sct_landmarks, max_pred, ttl])
        else: 
            ims.append([im, sct_landmarks, max_pred, ttl])

    # ims.append([im, sct_init, sct_pred,  ttl])

    plt.axis('off')
    if min_landmarks != None:
        if init_flag:
            plt.legend(['init', 'gt', 'min', 'max'], ncol = 4, loc = 'lower right', fontsize = 'small')
        else:
            plt.legend(['gt', 'min', 'max'], ncol = 4, loc = 'lower right', fontsize = 'small')

    else: 
        if init_flag:
            plt.legend(['init', 'gt', 'pred'], ncol = 3, loc = 'lower right', fontsize = 'small')
        else: 
            plt.legend(['groundtruth', 'estimation'], ncol = 3, loc = 'lower right', fontsize = 'small')

    return ims


def plot_single_gt_frame(sample):
    '''
    Plots a single frame and its ground truth where sample is returned dataset dictionary
    '''
    gt_landmarks_tmp = np.transpose(list(sample['landmarks']))
    gt_landmarks = np.flip(gt_landmarks_tmp, axis=1)

    if 'frame' in sample:
        image = sample['frame'][0]
    else: # for histogram event representation in image space
        image = torch.sum(sample['event_representation'], dim=0)
        image = norm_image(image.numpy())

    plt.figure()
    plt.imshow(image)
    plt.plot(gt_landmarks[:, 1], gt_landmarks[:, 0], '.', c='red', label='gt')
    plt.axis('off')
    plt.legend()


def plot_batch_from_dataloader(dataloader, batch_id):
    '''
    Test whether dataloader properly shuffles, visualizes the images in given batch number.
    '''
    for i_batch, sample_batched in enumerate(dataloader):
        print('Batch id: ', i_batch, sample_batched['frame'].size(), sample_batched['label'].size())

        # observe batch samples for batch_id and stop.
        if i_batch == batch_id:
            plt.figure()
            plot_labeled_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

def plot_samples_from_dataset(sq, skip, dataset):
    '''
    Visualize consecutive of images from the dataset

    :param sq: number of images displayed in square subplot for one side
    :param skip: how many consequent images to skip with each iter
    :return: displays 'sq x sq' consequent images from dataset
    '''

    fig, axs = plt.subplots(sq, sq)
    fig.suptitle('Samples from dataset')

    k = 0
    for j in range(sq):
        for i in range(sq):
            sample = dataset[j * sq + i + k]
            print('sample no:', j * sq + i + k, sample['frame'].shape, sample['label'].shape)
            blurred_label = gaussian_blur_label(sample['label'].float())
            axs[j, i].axis('off')
            # axs[j, i].set(xlabel='Camera #{}'.format(i), ylabel='Frame #{}'.format(j))
            axs[j, i].imshow(torch.sum(blurred_label, axis=0))
            axs[j, i].imshow(sample['frame'][0], cmap='gray', alpha=0.2)
            k += skip

    plt.show()

if __name__ == '__main__':
    from dataset.provider import get_dataset
    from path_locs import *
    from torch.utils.data import DataLoader
    from model import Net

    model = Net()
    model

    dataset_provider = get_dataset(path_dir, P_mat_dir)

    dhp19_dataset_train = dataset_provider.get_train_dataset()
    dhp19_dataset_test = dataset_provider.get_test_dataset()
    dhp19_dataset_val = dataset_provider.get_val_dataset()

    params = {'batch_size': 8,
              'shuffle': True,
              'num_workers': 0}

    train_loader = DataLoader(dhp19_dataset_train, **params)
    test_loader = DataLoader(dhp19_dataset_test, **params)
    val_loader = DataLoader(dhp19_dataset_val, **params)

    for i in range(1):
        for i_batch, sample_batched in enumerate(train_loader):
            inputs = sample_batched['frame']
            labels = sample_batched['label']

            output = model(inputs.float())
            plot_batch_label_pred_tb(sample_batched, output)
            plt.show()

            plot_labeled_batch(sample_batched)
            plt.show()

            if i_batch == 0:
                break
    
    
    