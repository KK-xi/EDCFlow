import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

from model.aggregate import SmallMotionEncoder, MotionFeatureEncoder
from model.Attention import SE, MSE
from model.transformer import FeatureAttention

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class DEPTHWISECONV3D(nn.Module):
    def __init__(self,in_ch,out_ch, kernel=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(DEPTHWISECONV3D, self).__init__()
        self.depth_conv = nn.Conv3d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel,
                                    stride=stride,
                                    padding=padding,
                                    groups=in_ch)
        self.point_conv = nn.Conv3d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class MEModule2(nn.Module):
    """ Motion exciation module with STE & CT exciation

    :param reduction=16
    :param n_segment=8/16
    """

    def __init__(self, channel, hdim=128, cdim=64, reduction=16, n_segment=6):
        super(MEModule2, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel // reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel // reduction,
            out_channels=self.channel // reduction,
            kernel_size=3,
            padding=1,
            groups=channel // reduction,
            bias=False)

        self.convf1 = nn.Sequential(
            DEPTHWISECONV3D(channel // reduction, channel * 2, (n_segment - 1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel * 2, channel, 1, 1, 0),)
        self.convf2 = nn.Sequential(
            DEPTHWISECONV3D(channel // reduction, channel * 2, (n_segment - 4, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel * 2, channel, 1, 1, 0),)
        self.convf3 = nn.Sequential(
            DEPTHWISECONV3D(channel // reduction, channel * 2, (1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel * 2, channel, 1, 1, 0),)

        self.mse = MSE(channel, M=3, r=2)
        self.convff = nn.Sequential(
            nn.Conv2d(channel * 3, channel * 3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 3, channel, 1, 1, 0),)

        self.SKAttn = SE(features=channel + channel, M=2, r=2)
        self.update = SepConvGRU(hidden_dim=hdim, input_dim=channel+channel+cdim)
        self.motion_enc = SmallMotionEncoder(corr_level=1, corr_radius=3)

        self.flow_head = nn.Sequential(nn.Conv2d(hdim, 128, 3, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 2, 3, padding=1))
        self.mask = nn.Sequential(
            nn.Conv2d(hdim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4 * 4 * 9, 1, padding=0))

    def forward(self, x, flow, net0, inp, corr):
        n, t, c, h, w = x.size()
        x = x.reshape(n*t, c, h, w)
        bottleneck = self.conv1(x)  # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w

        # t feature -1
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])
        t_fea, __ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)
        conv_bottleneck = self.conv2(bottleneck)
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        __, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)

        diff_fea = tPlusone_fea - t_fea  # diff
        diff_fea = rearrange(diff_fea, 'b t c h w -> b c t h w', b=n, h=h, w=w, t=self.n_segment - 1)
        out_difffea = diff_fea
        hd1 = self.convf1(diff_fea)
        hd1 = hd1.reshape(n, c, h, w)

        # t feature -2
        t_fea = reshape_bottleneck[:, ::2].split([self.n_segment - 4, 1], dim=1)[0]
        tPlusone_fea = reshape_conv_bottleneck[:, ::2].split([1, self.n_segment - 4], dim=1)[-1]

        diff_fea = tPlusone_fea - t_fea  # diff
        diff_fea = rearrange(diff_fea, 'b t c h w -> b c t h w', b=n, h=h, w=w, t=self.n_segment-4)
        hd2 = self.convf2(diff_fea)
        hd2 = hd2.reshape(n, c, h, w)

        # t feature -3
        t_fea = reshape_bottleneck[:, ::5].split([1, 1], dim=1)[0]
        tPlusone_fea = reshape_conv_bottleneck[:, ::5].split([1, 1], dim=1)[-1]

        diff_fea = tPlusone_fea - t_fea  # diff
        diff_fea = rearrange(diff_fea, 'b t c h w -> b c t h w', b=n, h=h, w=w, t=1)
        hd3 = self.convf3(diff_fea)
        hd3 = hd3.reshape(n, c, h, w)

        ms_hd = torch.stack([hd1, hd2, hd3], 1) + self.mse(torch.stack([hd1, hd2, hd3], 1))
        ms_hd = ms_hd.reshape(n, -1, h, w)
        hd = self.convff(ms_hd)

        mf = self.motion_enc(flow, corr)
        corr_mf = mf
        mf = torch.cat([hd, mf], 1) + self.SKAttn(torch.cat([hd, mf], 1))
        mf = mf.reshape(n, -1, h, w)

        inp = torch.cat([mf, inp], 1)
        net0 = self.update(net0, inp)

        dtflow = self.flow_head(net0)
        mask = self.mask(net0)

        return dtflow, mask, net0, [hd1, hd2, hd3], hd, corr_mf, mf,out_difffea, bottleneck.reshape(n, t, -1, h, w), conv_bottleneck.reshape(n, t, -1, h, w)