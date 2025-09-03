import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import ExtractorF, ExtractorC
from model.corr import CorrBlock
from model.util import coords_grid
from model.Motion_encoder import MEModule2

class EDCFlowNet(nn.Module):
    def __init__(self, args, input_bins=15):
        super(EDCFlowNet, self).__init__()
        self.args = args
        f_channel = 128
        self.split = 5
        self.corr_level = 1
        self.corr_radius = 3
        self.hdim=96
        self.cdim=64

        self.fnet = ExtractorF(input_channel=input_bins//self.split, outchannel=f_channel, norm='IN')
        self.cnet = ExtractorC(input_channel=input_bins//self.split + input_bins, outchannel=self.hdim+self.cdim, norm='BN')
        self.me = MEModule2(channel=64, hdim=96, cdim=64, reduction=1, n_segment=self.split + 1)

    def upsample_flow(self, flow, mask=None, scale=8):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        if mask is None:
            up_flow = F.interpolate(flow, scale_factor=scale,
                                    mode='bilinear', align_corners=False) * scale
        else:
            N, _, H, W = flow.shape
            mask = mask.view(N, 1, 9, scale, scale, H, W)
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(scale * flow, [3, 3], padding=1)
            up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

            up_flow = torch.sum(mask * up_flow, dim=2)
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
            up_flow = up_flow.reshape(N, 2, scale * H, scale * W)
        return up_flow

    @staticmethod
    def create_identity_grid(H, W, device):
        i, j = map(lambda x: x.float(), torch.meshgrid(
            [torch.arange(0, H), torch.arange(0, W)], indexing='ij'))
        return torch.stack([j, i], dim=-1).to(device)

    def warp_tensor(self, fmaps, flow):
        # raw: [T, N, C, H, W]
        N, T, C, H, W = fmaps.shape
        warp_tensor = torch.zeros_like(fmaps)
        warp_tensor[:, 0] = fmaps[:, 0]
        identity_grid = self.create_identity_grid(H, W, fmaps.device)
        for t in range(1, T):
            delta_p = flow * t / (T - 1)
            sampling_grid = identity_grid + torch.movedim(delta_p, 1, -1)
            sampling_grid[..., 0] = sampling_grid[..., 0] / (W - 1) * 2 - 1
            sampling_grid[..., 1] = sampling_grid[..., 1] / (H - 1) * 2 - 1
            warp_tensor[:, t,] = F.grid_sample(fmaps[:, t, ], sampling_grid, align_corners=False)

        return warp_tensor

    def forward(self, x1, x2):
        # x1, x2 = data['voxel_grid_prev'], data['voxel_grid_curr']
        b, _, h, w = x2.shape

        #Feature maps [f_0 :: f_i :: f_g]
        voxels = x2.chunk(self.split, dim=1)  # split, b, num_bins//split, h, w
        voxelref = x1.chunk(self.split, dim=1)[-1]  # 1, b, num_bins//split, h, w
        voxels = (voxelref,) + voxels  # [group+1] elements

        fmaps, fmaps_4 = self.fnet(voxels) # Tuple(f0, f1, ..., f_g)

        # Context map [net, inp]
        hdim, cdim = self.hdim, self.cdim
        cmap = self.cnet(torch.cat(voxels, dim=1))  # b, (split+1)num_bins//split, h, w
        net, inp4 = torch.split(cmap, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp4 = torch.relu(inp4)

        coords0 = coords_grid(b, h//4, w//4, device=cmap.device)
        coords1 = coords_grid(b, h//4, w//4, device=cmap.device)

        #MidCorr
        corr_fn = CorrBlock(fmaps[0], fmaps[-1], num_levels=self.corr_level,radius=self.corr_radius)  # [c01,c02,...,c05]

        flow_predictions = []
        for iter in range(self.args.iters):
            coords1 = coords1.detach()
            down_coords1 = F.interpolate(coords1, scale_factor=1/2, mode="bilinear", align_corners=False)/2
            corr_map = corr_fn(down_coords1)
            corr_map = F.interpolate(corr_map, scale_factor=2, mode='bilinear', align_corners=False)
            flow = coords1 - coords0
            warp_maps = self.warp_tensor(torch.stack(fmaps_4, 1), flow)
            delt_flow, mask, net = self.me(warp_maps, flow, net, inp4, corr_map)
            coords1 = coords1 + delt_flow

            flow_predictions.append(self.upsample_flow((coords1 - coords0), mask, scale=4))

        batch = dict(
            flow_preds=flow_predictions,  # flow sequence of the last time step: B, ietrs, 2, H, W
            flow_init=None,
        )
        return batch
