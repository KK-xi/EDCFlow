import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import os


scale=1
from Visualization.viz_utils import *


def viz_results(args, output, voxel_grid_prev, voxel_grid_curr, img1, iter):

    save_dir = os.path.join('./'+args.arch, 'iter_'+str(iter))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img1 = img1[0].permute(1, 2, 0).cpu().detach().numpy()   # H, W, 3
    f = os.path.join(save_dir, 'input_Image1.png')
    cv2.imwrite(f, img1)

    prev_voxel = voxel_grid_prev[:, -1].unsqueeze(1)
    viz_voxel_grid = torch.cat([prev_voxel, voxel_grid_curr], 1)[0].unsqueeze(1)  #  T+1, 1, H, W
    T = viz_voxel_grid.shape[0]
    for i in range(T):
        voxel_grid = viz_voxel_grid[i].cpu().transpose(0, 1).sum(1).detach().numpy()  # H, W
        f = os.path.join(save_dir, 'input_event_frame_{}.png'.format(i))
        plt.imsave(f, voxel_grid, cmap='gray')

    sequence_flow = (output['flow_preds'][0, -1]).unsqueeze(0).cpu().detach().numpy()   # t_steps = num_bins
    #--------------------------------------------------------------------------------------#
    x_flow = cv2.resize(np.array(sequence_flow[0, 0, :, :]), (scale * 640, scale * 480),
                        interpolation=cv2.INTER_LINEAR)
    y_flow = cv2.resize(np.array(sequence_flow[0, 1, :, :]), (scale * 640, scale * 480),
                        interpolation=cv2.INTER_LINEAR)
    flow_rgb = flow_to_image(x_flow, y_flow)
    flow_rgb = drawImageTitle(cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2RGB), 'output-flow', (0, 0, 0))
    f = os.path.join(save_dir, 'flow_pred.png')
    cv2.imwrite(f, flow_rgb)
    # --------------------------------------------------------------------------------------#