import os
import argparse

class ArgParser:
    def __init__(self):

        self.args = None
        self.parse = argparse.ArgumentParser(description='EDCFlowNet')
        self.parse.add_argument('--arch', type=str, default='EDCFlow')

        # net setting
        self.parse.add_argument("--loss_gamma", type=float, default=0.8, help="")
        self.parse.add_argument('--iters', type=int, default=6, help="iters from low level to higher")

        # training setting
        self.parse.add_argument('--num_epochs', type=int, default=100)  # for DSEC
        self.parse.add_argument('--checkpoint_root', type=str, default='./checkpoints')
        self.parse.add_argument('--save_interval', default=5, type=int, metavar='N',
                            help='Save model every \'save interval\' epochs ')
        self.parse.add_argument('--lr', type=float, default=2e-4)

        self.parse.add_argument('--dropout', default=False)
        self.parse.add_argument('--mixed_precision', default=False)
        self.parse.add_argument("--clip", type=float, default=1.0)

        # testing setting
        self.parse.add_argument('-e', '--evaluate', dest='evaluate', default=False,
                            help='evaluate model on validation set')
        self.parse.add_argument('--evaluate_interval', default=1, type=int, metavar='N',
                            help='Evaluate every \'evaluate interval\' epochs ')
        self.parse.add_argument('--viz_interval', default=2, type=int, metavar='N',
                                help='Evaluate visiualization every \'evaluate interval\' epochs ')
        self.parse.add_argument('--render', dest='render', default=False,
                            help='evaluate model on validation set')

        # datasets setting
        self.parse.add_argument('--data_root', type=str,
                                default=' ')
        self.parse.add_argument('--datasets', type=str, default='DSEC')
        self.parse.add_argument("--downsample_ratio", type=float, default=1, help="")
        self.parse.add_argument("--vertical_flip", type=bool, default=True, help="")
        self.parse.add_argument("--p_vflip", type=float, default=0.1, help="")
        self.parse.add_argument("--horizontal_flip", type=bool, default=True, help="")
        self.parse.add_argument("--p_hflip", type=float, default=0.5, help="")
        self.parse.add_argument("--random_crop", type=bool, default=True, help="")
        self.parse.add_argument('--crop_size', type=list, default=[288, 384])  # [288, 384] for dsec
        self.parse.add_argument("--concat_seq", type=bool, default=True, help="")
        self.parse.add_argument('--num_bins', '-nb', default=15)  # dsec: nb=15, mvsec: dt=4-->nb=9

        # dataloader setting
        self.parse.add_argument('--batch_size', type=int, default=3)
        self.parse.add_argument('--num_workers', type=int, default=8)
        self.parse.add_argument('--device', type=str, default='cuda:2')
        # model setting
        self.parse.add_argument('--grad_clip', type=float, default=1)

    def parser(self):
        self.args = self.parse.parse_args()
        return self.args
