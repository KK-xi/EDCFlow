import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


import numpy as np
import cv2
import torch
import torch.nn as nn
import re
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def apply_transform_to_field(sample, func, field_name):
    """
    Applies a function to a field of a sample.
    :param sample: a sample
    :param func: a function that takes a numpy array and returns a numpy array
    :param field_name: the name of the field to transform
    :return: the transformed sample
    """
    if isinstance(sample, dict):
        for key, value in sample.items():
            if bool(re.search(field_name, key)):
                sample[key] = func(value)

    if isinstance(sample, np.ndarray) or isinstance(sample, torch.Tensor):
        return func(sample)
    return sample


def apply_randomcrop_to_sample(sample, crop_size):
    """
    Applies a random crop to a sample.
    :param sample: a sample
    :param crop_size: the size of the crop
    :return: the cropped sample
    """
    i, j, h, w = RandomCrop.get_params(
        sample["voxel_grid_prev"], output_size=crop_size)
    keys_to_crop = ["voxel_grid_prev", "voxel_grid_curr",
                    "flow_gt_voxel_grid_prev", "flow_gt_voxel_grid_curr", "reverse_flow_gt_voxel_grid_prev", "reverse_flow_gt_voxel_grid_curr"]

    for key, value in sample.items():
        if key in keys_to_crop:
            if isinstance(value, torch.Tensor):
                sample[key] = TF.crop(value, i, j, h, w)
            elif isinstance(value, list) or isinstance(value, tuple):
                sample[key] = [TF.crop(v, i, j, h, w) for v in value]
    return sample


def downsample_spatial(x, factor):
    """
    Downsample a given tensor spatially by a factor.
    :param x: PyTorch tensor of shape [batch, num_bins, height, width]
    :param factor: downsampling factor
    :return: PyTorch tensor of shape [batch, num_bins, height/factor, width/factor]
    """
    assert (factor > 0), 'Factor must be positive!'

    assert (x.shape[-1] %
            factor == 0), 'Width of x must be divisible by factor!'
    assert (x.shape[-2] %
            factor == 0), 'Height of x must be divisible by factor!'

    return nn.AvgPool2d(kernel_size=factor, stride=factor)(x)


def downsample_spatial_mask(x, factor):
    """
    Downsample a given mask (boolean) spatially by a factor.
    :param x: PyTorch tensor of shape [batch, num_bins, height, width]
    :param factor: downsampling factor
    :return: PyTorch tensor of shape [batch, num_bins, height/factor, width/factor]
    """
    assert (factor > 0), 'Factor must be positive!'

    assert (x.shape[-1] %
            factor == 0), 'Width of x must be divisible by factor!'
    assert (x.shape[-2] %
            factor == 0), 'Height of x must be divisible by factor!'

    return nn.AvgPool2d(kernel_size=factor, stride=factor)(x.float()) >= 0.5


class Augmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.4, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 0.5]
        flow0 = flow[valid >= 0.5]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, voxel1, voxel2, flow, valid):
        ht, wd = voxel2.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            voxel1 = cv2.resize(voxel1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            voxel2 = cv2.resize(voxel2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
            # print('Resized:', voxel1.shape, voxel2.shape, flow.shape, valid.shape)

        margin_y = int(round(65 * scale_y))  # downside
        margin_x = int(round(35 * scale_x))  # leftside

        y0 = np.random.randint(0, voxel2.shape[0] - self.crop_size[0] - margin_y)
        x0 = np.random.randint(margin_x, voxel2.shape[1] - self.crop_size[1])

        y0 = np.clip(y0, 0, voxel2.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, voxel2.shape[1] - self.crop_size[1])

        voxel1 = voxel1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        voxel2 = voxel2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                voxel1 = voxel1[:, ::-1]
                voxel2 = voxel2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                voxel1 = voxel1[::-1, :]
                voxel2 = voxel2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                valid = valid[::-1, :]
        return voxel1, voxel2, flow, valid

    def __call__(self, voxel1, voxel2, flow, valid):
        voxel1, voxel2, flow, valid = self.spatial_transform(voxel1, voxel2, flow, valid)
        voxel1 = np.ascontiguousarray(voxel1)
        voxel2 = np.ascontiguousarray(voxel2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return voxel1, voxel2, flow, valid

