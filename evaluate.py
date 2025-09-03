import imageio

imageio.plugins.freeimage.download()
# import imageio.v3 as iio
import os
from torch.utils.data import DataLoader
import glob
import random
from tqdm import tqdm
import torch
import time
import numpy as np
import argparse
from pathlib import Path
from typing import Dict

from model.edcflow import EDCFlowNet
from dataloader.dsec_loader import DSECfull

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def is_string_swiss(input_str: str) -> bool:
    is_swiss = False
    is_swiss |= 'thun_' in input_str
    is_swiss |= 'interlaken_' in input_str
    is_swiss |= 'zurich_city_' in input_str
    return is_swiss


def files_per_sequence(flow_timestamps_dir: Path) -> Dict[str, list]:
    out_dict = dict()
    for entry in flow_timestamps_dir.iterdir():
        assert entry.is_file()
        assert entry.suffix == '.csv', entry.suffix
        assert is_string_swiss(entry.stem), entry.stem
        data = np.loadtxt(entry, dtype=np.int64, delimiter=', ', comments='#')
        assert data.ndim == 2, data.ndim
        out_dict[entry.stem] = data[:, -1]
    return out_dict


class ArgParser:
    def __init__(self):

        self.args = None
        self.parse = argparse.ArgumentParser(description='EDCFlowNet')
        self.parse.add_argument('--arch', type=str, default='EDCFlow')

        # net setting
        self.parse.add_argument('--iters', type=int, default=6, help="iters from low level to higher")

        # training setting
        self.parse.add_argument('--checkpoint_root', type=str, default='./checkpoints')
        self.parse.add_argument('--save_interval', default=5, type=int, metavar='N',
                            help='Save model every \'save interval\' epochs ')

        # testing setting
        self.parse.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                            help='evaluate model on validation set')
        self.parse.add_argument('--evaluate_interval', default=1, type=int, metavar='N',
                            help='Evaluate every \'evaluate interval\' epochs ')
        self.parse.add_argument('--viz_interval', default=2, type=int, metavar='N',
                                help='Evaluate visiualization every \'evaluate interval\' epochs ')
        self.parse.add_argument('--render', dest='render', default=False,
                            help='evaluate model on validation set')
        self.parse.add_argument('--test_save', type=str, default='./EDCFlow')  # !

        # datasets setting
        self.parse.add_argument('--data_root', type=str,
                                default='')
        self.parse.add_argument('--datasets', type=str, default='DSEC')
        self.parse.add_argument("--concat_seq", type=bool, default=True, help="")
        self.parse.add_argument('--num_bins', '-nb', default=15)  # dsec: nb=15, mvsec: dt=4-->nb=9

        # dataloader setting
        self.parse.add_argument('--batch_size', type=int, default=1)
        self.parse.add_argument('--num_workers', type=int, default=4)

    def parser(self):
        self.args = self.parse.parse_args()
        return self.args


class Evaluate:
    def __init__(self, args):

        self.args = args

        time = '10-27-14:04'
        model_name = 'EDCFlow'  # !
        save_name = 'final_checkpoint.pth.tar'
        ckpt = os.path.join(args.checkpoint_root, time, model_name, save_name)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device='cpu'
        self.device = device

        self.model = EDCFlowNet(self.args)
        self.model.load_state_dict(torch.load(ckpt, map_location='cuda:0')['state_dict'])
        self.model = self.model.to(device)

        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print("Total number of parameters : %.4f M" % (num_params / 1e6))

        if not os.path.exists(args.test_save):
            os.mkdir(args.test_save)

        # data loader
        test_set = DSECfull(args, augment=False, instance='test')
        self.test_loader = DataLoader(test_set, batch_size=1, pin_memory=True,
                                       num_workers=4, shuffle=False)
        print('test_loader done!')

    def evaluate(self):
        self.model.eval()
        iters = 0

        time_list = []
        bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for i, data in bar:
            city = data['seq_name'][0][0]
            file_ind = data['file_idx'][0][0]
            submission = data['submission'][0][0]

            if submission:
                voxel_grid_curr = data['voxel_grid_curr']
                voxel_grid_prev = data['voxel_grid_prev']
                voxel_grid_curr = voxel_grid_curr.to(self.device)
                voxel_grid_prev = voxel_grid_prev.to(self.device)

                # compute output
                start = time.time()
                output = self.model(voxel_grid_prev, voxel_grid_curr)
                end = time.time()
                time_list.append((end - start) * 1000)

                last_flow_list = (output['flow_preds'])
                flo = last_flow_list[-1][0].permute(1, 2, 0).cpu().numpy()  # h, w, 2
                uv = flo * 128.0 + 2 ** 15
                valid = np.ones([uv.shape[0], uv.shape[1], 1])
                uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)

                city = os.path.join(args.test_save, city)
                if not os.path.exists(city):
                    os.mkdir(city)

                path_to_file = os.path.join(city, 'seq_{:06d}'.format(file_ind) + '.png')
                # print(path_to_file)
                imageio.imwrite(path_to_file, uv, format='PNG-FI')

        avg_time = sum(time_list) / len(time_list)
        print(f'Time: {avg_time} ms.')
        print('Done!')

        return None

    def run_network(self):
        with torch.no_grad():
            evaluate_out = self.evaluate()


if __name__ == '__main__':
    argparser = ArgParser()
    args = argparser.parser()
    set_seed(1)

    evaluator = Evaluate(args)
    evaluator.run_network()