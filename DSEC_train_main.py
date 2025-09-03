import datetime
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import random

from config.params_config import ArgParser
from model.edcflow import EDCFlowNet
from dataloader.dsec_loader import DSECfull

MAX_FLOW = 400
SUM_FREQ = 200
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()#b,h,w
    valid = (valid >= 0.5) & (mag < max_flow) # b,1,h,w

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


class Trainer:
    def __init__(self, args):
        self.args = args

        # device_ids = [2]
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = EDCFlowNet(self.args)
        # self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.model = self.model.to(device)

        num_params = 0
        for param in self.model.parameters():
            num_params += param.numel()
        print("Total number of parameters : %.4f M" % (num_params / 1e6))

        # data loader
        train_set = DSECfull(args, augment=True, instance='train')
        self.train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                                       num_workers=8, shuffle=True, drop_last=True)
        print('train_loader done!')

        # Optimizer and scheduler for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=args.lr,
            steps_per_epoch=len(self.train_loader),
            epochs=args.num_epochs,
            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear',
        )

        save_path = '{}'.format(args.arch)
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp, save_path)
        self.save_path = os.path.join(args.checkpoint_root, save_path)

        print('=> Everything will be saved to {}'.format(self.save_path))

        if not os.path.exists(self.save_path) and not args.evaluate:
            os.makedirs(self.save_path)

        self.writer = SummaryWriter(os.path.join(self.save_path, 'summary'))

        self.epoch = 0
        self.total_steps = 0
        self.num_steps = args.num_epochs*len(self.train_loader)
        self.batches_seen = 0
        self.samples_seen = 0

    def train(self):
        self.model.train()
        num_steps = args.num_epochs*len(self.train_loader)
        loss_sum, EPE, PX1, PX3, PX5, iters = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        best_EPE = -1

        total_steps = 0
        epoch = 0
        while epoch < self.args.num_epochs:

            bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=60)
            for index, data in bar:
                voxel_grid_curr, voxel_grid_prev, gt_flow, flow_valid = \
                    data['voxel_grid_curr'], data['voxel_grid_prev'], data['gt_flow'], data['flow_valid']
                voxel_grid_curr, voxel_grid_prev, gt_flow, flow_valid = voxel_grid_curr.to(self.device), \
                    voxel_grid_prev.to(self.device), \
                    gt_flow.to(self.device), \
                    flow_valid.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(voxel_grid_prev, voxel_grid_curr)
                flow_list = output['flow_preds']
                flow_loss, loss_metrics = sequence_loss(flow_list, gt_flow, flow_valid, gamma=0.8,
                                                        max_flow=MAX_FLOW)
                flow_loss.backward()
                self.optimizer.step()
                total_steps += 1

                self.scheduler.step()

                bar.set_description(f'Step: {total_steps}/{num_steps}')
                loss_sum += flow_loss.item()
                EPE += (loss_metrics["epe"])
                PX1 += (loss_metrics["1px"])
                PX3 += (loss_metrics["3px"])
                PX5 += (loss_metrics["5px"])
                iters += 1

                if iters % SUM_FREQ == 0:
                    mean_loss = loss_sum / SUM_FREQ
                    mean_EPE = EPE / SUM_FREQ
                    mean_PX1 = PX1 / SUM_FREQ
                    mean_PX3 = PX3 / SUM_FREQ
                    mean_PX5 = PX5 / SUM_FREQ

                    loss_sum, EPE, PX1, PX3, PX5, iters = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                    self.writer.add_scalar('train/mean_train_loss', mean_loss, total_steps)
                    self.writer.add_scalar('train/mean_EPE', mean_EPE, total_steps)
                    self.writer.add_scalar('train/mean_1px', mean_PX1, total_steps)
                    self.writer.add_scalar('train/mean_3px', mean_PX3, total_steps)
                    self.writer.add_scalar('train/mean_5px', mean_PX5, total_steps)
                    self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], total_steps)

                    print("Trianing on {} with {:02d}/{:02d} Loss: {:2.6f} EPE: {:2.6f}".format(
                        args.datasets, total_steps, self.num_steps, mean_loss, mean_EPE))

                if total_steps >= self.num_steps - 10000 and total_steps % 5000 == 0:
                    filename = f'{total_steps}.pth.tar'
                    torch.save({
                        'steps': total_steps,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                    }, os.path.join(self.save_path, filename))

            epoch += 1

        filename = 'final_checkpoint.pth.tar'
        torch.save({
            'steps': total_steps,
            'arch': args.arch,
            'state_dict': self.model.state_dict(),
        }, os.path.join(self.save_path, filename))

        print('The best EPE at: {}'.format(best_EPE))

        return filename



if __name__ == '__main__':
    argparser = ArgParser()
    args = argparser.parser()
    set_seed(1)

    trainer = Trainer(args)
    trainer.train()