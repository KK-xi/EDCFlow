import numpy as np
import torch
import torch.utils.data as data

import random
import os
import glob
from torchvision import transforms as tf

from dataloader.augment import ( downsample_spatial, downsample_spatial_mask,
                               apply_transform_to_field, apply_randomcrop_to_sample)


class DSECfull(data.Dataset):
    def __init__(self, args, augment=True, instance='train'):
        super(DSECfull, self).__init__()
        self.args = args
        self.root = args.data_root
        self.instance = instance

        self.files = []
        if instance is 'train':
            self.files += glob.glob(os.path.join(self.root, 'train_data', '*', 'seq_*.npz'))
        else:
            self.files += glob.glob(os.path.join(self.root, 'test_data', '*', 'seq_*.npz'))
        self.files.sort()

        self.transforms = None
        if (augment is True) and (instance is 'train'):
            # Prepare transform list
            transforms = dict()
            if args.downsample_ratio > 1:
                transforms['(?<!flow_gt_)event_volume'] = lambda sample: downsample_spatial(
                    sample, args.downsample_ratio)
                transforms['flow_gt'] = lambda sample: [downsample_spatial(
                    sample[0], args.downsample_ratio) / args.downsample_ratio,
                                                        downsample_spatial_mask(sample[1], args.downsample_ratio)]
                # TODO: Flow gt mask handle downsample on flags
            if args.horizontal_flip:
                p_hflip = args.p_hflip
                assert p_hflip >= 0 and p_hflip <= 1
                # from torchvision.transforms import RandomHorizontalFlip
                # ignore probability of hflip for now, perform when 'hflip' key exists in transforms dict
                transforms['hflip'] = None
            if args.vertical_flip:
                p_vflip = args.p_vflip
                assert p_vflip >= 0 and p_vflip <= 1
                transforms['vflip'] = p_vflip
            if args.random_crop:
                crop_size = args.crop_size
                transforms['randomcrop'] = crop_size
            self.transforms = transforms

    def __getitem__(self, index):

        file_name = self.files[index]
        file = np.load(file_name, allow_pickle=True)

        if self.instance is 'train':

            voxel_grid_curr = file['voxel_grid_curr']  # B, H, W
            voxel_grid_prev = file['voxel_grid_prev']  # B, H, W

            gt_flow = file['flow_gt_voxel_grid_curr']  # 2 H, W
            flow_valid = file['flow_gt_voxel_grid_curr_valid_mask']  # 1, H, W

            voxel_grid_curr = torch.from_numpy(voxel_grid_curr)
            voxel_grid_prev = torch.from_numpy(voxel_grid_prev)
            gt_flow = torch.from_numpy(gt_flow)
            flow_valid = torch.from_numpy(flow_valid)
            flow_mask = [gt_flow, flow_valid]

            sample = dict(voxel_grid_curr=voxel_grid_curr, voxel_grid_prev=voxel_grid_prev, flow_gt_voxel_grid_curr=flow_mask)

            if self.transforms is not None:
                for key_t, transform in self.transforms.items():
                    if key_t == "hflip":
                        if random.random() > 0.5:
                            for key in sample:
                                if isinstance(sample[key], torch.Tensor):
                                    sample[key] = tf.functional.hflip(sample[key])
                                if key.startswith("flow_gt"):
                                    sample[key] = [tf.functional.hflip(
                                        mask) for mask in sample[key]]
                                    sample[key][0][0, :] = -sample[key][0][0, :]
                    elif key_t == "vflip":
                        if random.random() < transform:
                            for key in sample:
                                if isinstance(sample[key], torch.Tensor):
                                    sample[key] = tf.functional.vflip(sample[key])
                                if key.startswith("flow_gt"):
                                    sample[key] = [tf.functional.vflip(
                                        mask) for mask in sample[key]]
                                    sample[key][0][1, :] = -sample[key][0][1, :]
                    elif key_t == "randomcrop":
                        apply_randomcrop_to_sample(sample, crop_size=transform)
                    else:
                        apply_transform_to_field(sample, transform, key_t)

            voxel_grid_curr = sample["voxel_grid_curr"].float()
            voxel_grid_prev = sample["voxel_grid_prev"].float()
            gt_flow = sample["flow_gt_voxel_grid_curr"][0].float()
            flow_valid = sample["flow_gt_voxel_grid_curr"][1].float()

            batch = dict(voxel_grid_curr=voxel_grid_curr, voxel_grid_prev=voxel_grid_prev,
                         gt_flow=gt_flow, flow_valid=flow_valid)
        else:
            voxel_grid_curr = file['voxel_grid_curr']  # B, H, W
            voxel_grid_prev = file['voxel_grid_prev']  # B, H, W
            voxel_grid_curr = torch.from_numpy(voxel_grid_curr).float()
            voxel_grid_prev = torch.from_numpy(voxel_grid_prev).float()

            submission = [file['save_submission']]
            file_idx = [file['file_index']]
            seq_name = [file['seq_name']]

            batch = dict(voxel_grid_curr=voxel_grid_curr, voxel_grid_prev=voxel_grid_prev,
                         submission=submission, file_idx=file_idx, seq_name=seq_name)

        return batch

    def __len__(self):
        return len(self.files)


def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D


def make_data_loader(phase, batch_size, num_workers):
    dset = DSECfull(phase)
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True)
    return loader


if __name__ == '__main__':
    dset = DSECfull('test')
    print(len(dset))
    v1, v2, flow, valid = dset[0]
    print(v1.shape, v2.shape, flow.shape, valid.shape)
