import pdb
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision.transforms.functional import gaussian_blur

def get_pred_landmarks(pred_batch, device=None):
    '''
    Get joint location positions in pixel space from prediction.
    :batched_sample: batch of size (num_batch, num_joints, img_height, img_width)

    :return: array of dimension [batch_size, 2, num_joints]
    '''

    assert pred_batch.dim() == 4

    if device == 'cpu':
        pred_batch = pred_batch.to('cpu')
    (batch_size, num_joints, img_height, img_width) = pred_batch.shape
    flat = pred_batch.reshape((batch_size, num_joints, -1))
    max_indices = torch.argmax(flat, dim=-1)

    # map flattened indices to 2d indices
    y_coord = torch.div(max_indices, img_width, rounding_mode='floor')
    x_coord = max_indices % img_width

    y_coord = y_coord.reshape(batch_size, 1, num_joints)
    x_coord = x_coord.reshape(batch_size, 1, num_joints)

    pred_landmarks = torch.concat((x_coord, y_coord), dim=1)
    return pred_landmarks

def get_min_landmarks(pred_batch, device=None):
    '''
    Get minimum pixel values points for each joint heatmap.
    :batched_sample: batch of size (num_batch, num_joints, img_height, img_width)

    :return: array of dimension [batch_size, 2, num_joints]
    '''

    assert pred_batch.dim() == 4

    if device == 'cpu':
        pred_batch = pred_batch.to('cpu')
    (batch_size, num_joints, img_height, img_width) = pred_batch.shape
    flat = pred_batch.reshape((batch_size, num_joints, -1))
    min_indices = torch.argmin(flat, dim=-1)

    # map flattened indices to 2d indices
    y_coord = torch.div(min_indices, img_width, rounding_mode='floor')
    x_coord = min_indices % img_width

    y_coord = y_coord.reshape(batch_size, 1, num_joints)
    x_coord = x_coord.reshape(batch_size, 1, num_joints)

    pred_landmarks = torch.concat((x_coord, y_coord), dim=1)
    return pred_landmarks

def calc_mpjpe(pred_landmarks, target):

    assert pred_landmarks.dim() == 3
    assert target.dim() == 3

    dist_2d = torch.linalg.norm((pred_landmarks - target), axis=1)
    mpjpe_per_sample = torch.nanmean(dist_2d, axis=-1)
    mpjpe = torch.nanmean(mpjpe_per_sample) # batch average
    return mpjpe

def norm_image_tensor(img_tensor):
    '''
    # todo do for 4D 
    Image normalizer for raw event counts in tensor space
    :param img: image tensor of shape [batch_size, image_h, image_w]
    :return: batch of normalized images
    '''
    (_, m, n) = img_tensor.shape
    sum_img = torch.sum(img_tensor, dim=(1,2))
    img_mask = img_tensor > 0
    count_img = torch.sum(img_mask, dim=(1,2))
    mean_img = torch.div(sum_img, count_img)
    masked_img = img_tensor * img_mask.int().float()
    var_img = torch.var(masked_img, dim=(1,2))
    sig_img = torch.sqrt(var_img)

    sig_img_mask = sig_img < 0.1 / 255

    if sig_img_mask.any() == True:
        sig_img[sig_img_mask] = 0.1 / 255

    numSDevs = 3.0
    polarity = True
    meanGrey = 0
    range_m = numSDevs * sig_img
    halfrange = 0
    rangenew = 255

    img_tensor= img_tensor * rangenew
    img_tensor = torch.div(img_tensor, torch.reshape(range_m, (range_m.shape[0], 1, 1)) , rounding_mode='floor')

    # make sure image values are within range
    img_tensor[img_tensor > rangenew] = rangenew
    img_tensor[img_tensor < 0] = 0

    return img_tensor


def norm_image(img):
    '''
    Image normalizer for raw event counts in pixel space
    :param img: total event count in pixel space
    :return: normalized image
    '''
    (m, n) = np.shape(img)
    sum_img = np.sum(np.sum(img))
    count_img = np.sum(np.sum(img > 0))
    mean_img = sum_img / count_img
    var_img = np.var(img[img > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    numSDevs = 3.0
    polarity = True
    meanGrey = 0
    range_m = numSDevs * sig_img
    halfrange = 0
    rangenew = 255

    for i_id in range(int(m)):
        for j_id in range(int(n)):
            l = img[i_id, j_id]
            if l == 0:
                img[i_id, j_id] = meanGrey
            if l != 0:
                f = (l + halfrange) * rangenew / range_m
                if f > rangenew:
                    f = rangenew

                if f < 0:
                    f = 0
                img[i_id, j_id] = np.floor(f)

    return img

def gaussian_blur_label(sample):
    """
    Blur labels from dataset.

    :sample: single label from dataset or a batch of labels from dataloader
             size [13,260,344] or [batch_size, 13, 260, 344]
    :return: gaussian blurred and normalized samples of same dimension

    """
    sigma = 2
    gaussian_kernel_size = int(((sigma - 0.8) / 0.3 + 1) * 2 + 1)
    image_h, image_w, num_joints = 260, 344, 13

    output_tmp = gaussian_blur(sample, gaussian_kernel_size, sigma)
    if sample.shape[0] == 13:
        output = torch.zeros((num_joints, image_h, image_w))

        max_tmp = torch.amax(output_tmp, dim=[1, 2])

    else:
        batch_size = sample.shape[0]
        output = torch.zeros((batch_size, num_joints, image_h, image_w))

        max_tmp = torch.amax(output_tmp, dim=[2, 3])

    mask = max_tmp != 0
    output[mask] = torch.divide(output_tmp[mask], max_tmp[mask].reshape(-1, 1, 1))

    return output

## FUNCTION TO COMPUTE LOSS ##
def my_mse_loss(output, target, sigma, device, no_blur=False):

    '''
    First do gaussian blur an normalization on the labels before computing the MSE loss. 
    This is done to make the learning easier. 

    We do it here instead of inside the dataloader because it's faster on GPU. 
    
    # default sigma value should be 2 (eg. for 256x256)
    # if donwsampled to half its size, sigma should be 1.1 (eg. for 128x128)
    '''
    
    assert output.dim() == 4
    assert target.dim() == 4
    
    batch_size, num_joints, image_h, image_w = output.shape

    if no_blur: 
        mse = torch.mean(torch.square(output - target))
    
    else:
        gaussian_kernel_size = int(((sigma - 0.8) / 0.3 + 1) * 2 + 1)

        target_blurred = gaussian_blur(target, gaussian_kernel_size, sigma).to(torch.float32)
        target_blurred_normalized = torch.zeros((batch_size, num_joints, image_h, image_w), device=device, dtype = torch.float32)

        max_tmp = torch.amax(target_blurred, dim=[2, 3])

        mask = max_tmp != 0
        target_blurred_normalized[mask] = torch.divide(target_blurred[mask], max_tmp[mask].reshape(-1, 1, 1))

        mse = torch.mean(torch.square(output - target_blurred_normalized))

    return mse

def blur_and_normalize(x, sigma, device):
    # default sigma value should be 2
    # if donwsampled to half its size, sigma should be 1.1
    batch_size, num_joints, image_h, image_w = x.shape

    x_blurred = gaussian_blur(x, kernel_size=11, sigma=sigma).to(torch.float32)
    
    x_final = torch.zeros((batch_size, num_joints, image_h, image_w), device=device, dtype = torch.float32)
    max_tmp = torch.amax(x_blurred, dim=[2, 3]).to(torch.float32)

    mask = max_tmp != 0
    x_final[mask] = torch.divide(x_blurred[mask], max_tmp[mask].reshape(-1, 1, 1))

    return x_final

# class for hooking layers to retrieve activations (ie. spikes)
class manage_hooks():
    '''
    For getting activations in NN layers. 

    1) Register forward hooks before calling the model. 
    2) Call the model which fills the activation dictionary. 
    3) After using the activations remove forward hooks so that the network doesn't keep track of activation layers.

    '''
    def __init__(self, model, hybrid=False, all_snn=False):
        self.model = model
        self.hybrid = hybrid
        self.all_snn = all_snn

        self.static_conv_hook = None
        self.down1_hook = None 
        self.down2_hook = None
        self.down3_hook = None
        self.residualBlock_hook = None
        self.up1_hook = None
        self.up2_hook = None 
        self.up3_hook = None
        
        self.activation = {}

    def _get_activation(self, name):
        def hook(model, input, output):
            if self.all_snn and (name == 'residualBlock' or name == 'residualBlock_snn'):
                output = output[0]
            self.activation[name] = output.detach()
        return hook

    def register_forward_hooks(self):
        if self.hybrid:
            self.static_conv_hook = self.model.static_conv_snn.register_forward_hook(self._get_activation('static_conv_snn'))
            self.down1_hook = self.model.down1_snn.register_forward_hook(self._get_activation('down1_snn'))
            self.down2_hook = self.model.down2_snn.register_forward_hook(self._get_activation('down2_snn'))
            self.down3_hook = self.model.down3_snn.register_forward_hook(self._get_activation('down3_snn'))
            self.residualBlock_hook = self.model.residualBlock_snn.register_forward_hook(self._get_activation('residualBlock_snn'))
            self.up1_hook = self.model.up1_snn.register_forward_hook(self._get_activation('up1_snn'))
            self.up2_hook = self.model.up2_snn.register_forward_hook(self._get_activation('up2_snn'))
            self.up3_hook = self.model.up3_snn.register_forward_hook(self._get_activation('up3_snn'))
        else: 
            self.static_conv_hook = self.model.static_conv.register_forward_hook(self._get_activation('static_conv'))
            self.down1_hook = self.model.down1.register_forward_hook(self._get_activation('down1'))
            self.down2_hook = self.model.down2.register_forward_hook(self._get_activation('down2'))
            self.down3_hook = self.model.down3.register_forward_hook(self._get_activation('down3'))
            self.residualBlock_hook = self.model.residualBlock.register_forward_hook(self._get_activation('residualBlock'))
            self.up1_hook = self.model.up1.register_forward_hook(self._get_activation('up1'))
            self.up2_hook = self.model.up2.register_forward_hook(self._get_activation('up2'))
            self.up3_hook = self.model.up3.register_forward_hook(self._get_activation('up3'))

        return self.activation

    def remove_forward_hooks(self):   
        print('hooks removed')  
        self.static_conv_hook.remove()
        self.down1_hook.remove()
        self.down2_hook.remove()
        self.down3_hook.remove()
        self.residualBlock_hook.remove()
        self.up1_hook.remove()
        self.up2_hook.remove()
        self.up3_hook.remove()

        self.activation.clear()
    