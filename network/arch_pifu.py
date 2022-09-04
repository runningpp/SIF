from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import network.hrnet as HRNET
import logging
import config


def othogonal(points, calibrations, transforms=None):
    '''
        Compute the orthogonal projections of 3D points into the image plane by given projection matrix
        :param points: [B, N,3] Tensor of 3D points
        :param calibrations: [B, 4, 4] Tensor of projection matrix
        :param transforms: [B, 2, 3] Tensor of image transform matrix
        :return: xyz: [B, n,3] Tensor of xyz coordinates in the image plane
        '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    points = points.permute(0,2,1)
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts.permute(0,2,1)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='xavier', gain=0.02):
        '''
        initializes network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                         or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class MLP(BaseNetwork):
    """
    MLP implemented using 2D convolution
    Neuron number: (257, 1024, 512, 256, 128, 1)
    """
    def __init__(self, in_channels=257, out_channels=1, bias=True, out_sigmoid=True, weight_norm=False):
        super(MLP, self).__init__()
        inter_channels = (1024, 512, 256, 128)
        norm_fn = lambda x: x
        if weight_norm:
            norm_fn = lambda x: nn.utils.weight_norm(x)

        self.conv0 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=in_channels, out_channels=inter_channels[0],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[0] + in_channels,
                              out_channels=inter_channels[1],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[1] + in_channels,
                              out_channels=inter_channels[2],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            norm_fn(nn.Conv2d(in_channels=inter_channels[2] + in_channels,
                              out_channels=inter_channels[3],
                              kernel_size=1, stride=1, padding=0, bias=bias)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        if out_sigmoid:
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=inter_channels[3], out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0, bias=bias),
                nn.Sigmoid()
            )
        else:
            self.conv4 = nn.Conv2d(in_channels=inter_channels[3], out_channels=out_channels,
                                   kernel_size=1, stride=1, padding=0, bias=bias)
        self.init_weights()

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(torch.cat([x, out], dim=1))
        out = self.conv2(torch.cat([x, out], dim=1))
        out = self.conv3(torch.cat([x, out], dim=1))
        out = self.conv4(out)
        return out

    def forward0(self, x):
        out = self.conv0(x)
        out = self.conv1(torch.cat([x, out], dim=1))
        out = self.conv2(torch.cat([x, out], dim=1))
        out = self.conv3(torch.cat([x, out], dim=1))
        return out

    def forward1(self, x, out):
        out = self.conv4(out)
        return out

class PIFuNet(BaseNetwork):

    def __init__(self,feat_dim=256,ortho=True):
        super(PIFuNet, self).__init__()
        # self.hg = hg.HourglassNet(4, 4, 128, 64)
        # self.mlp = MLP()
        self.feat_ch_2D = feat_dim
        self.ortho = ortho
        self.add_module('hrnet', HRNET.HRNetV2_W18_small_v2(
            inputMode=config.input_mode, numOutput=self.feat_ch_2D, normLayer=nn.BatchNorm2d))

        self.add_module('mlp', MLP(self.feat_ch_2D + 1, 1, weight_norm=False))

        logging.info('#trainable params of hourglass = %d' %
                     sum(p.numel() for p in self.hrnet.parameters() if p.requires_grad))

        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))

    def forward(self, img, pts,calibration,normal=None):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        pt_sdf_list = []

        if normal is not None:
            img = torch.cat([img, normal], dim=1)
            if self.ortho:
                pts = othogonal(pts,calibration)
            else:
                assert 1==2,print('no orthogoal')
        if self.training:

            img_feats = self.hrnet(img)

            h_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1)
            v_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1)

            grid_2d = torch.cat([h_grid, v_grid], dim=-1)

            z_feat = pts[:, :, 2].view(batch_size, 1, point_num, 1)
            z_feat = z_feat * (512 // 2) / 200
            for img_feat in img_feats:
                pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d,align_corners=True,
                                           mode='bilinear', padding_mode='border')
                del grid_2d,h_grid,v_grid
                pt_feat = torch.cat([pt_feat_2D, z_feat], dim=1)
                pt_output = self.mlp(pt_feat)  # shape = [batch_size, channels, point_num, 1]
                pt_output = pt_output.permute([0, 2, 3, 1])
                pt_sdf = pt_output.view([batch_size, point_num])
                pt_sdf_list.append(pt_sdf)
            return pt_sdf_list
        else:
            with torch.no_grad():
                img_feats = self.hrnet(img)

                img_feat = img_feats[-1]
                group_size = config.point_group_size
                group_num = (
                    (point_num // group_size) if (point_num % group_size == 0) else (point_num // group_size + 1))
                if (group_num == 0):
                    group_num = 1
                    group_size = point_num
                    print('forward: pts.shape[0]: %d' % point_num)
                    print('Total point group number: %d' % group_num)

                pt_sdf = []
                for gi in range(group_num):
                    # print('Processing point group %d ...' % gi)
                    start = gi * group_size
                    end = (gi + 1) * group_size
                    end = (end if (end <= point_num) else point_num)
                    group_size = end - start


                    pts_group = pts[:, start:end, :]
                    index_invalid = (pts_group[:, :, 0] >= 1) + (pts_group[:, :, 0] <= -1) + \
                                    (pts_group[:, :, 1] >= 1) + (pts_group[:, :, 1] <= -1)
                    pts_group[index_invalid] = -1

                    h_grid = pts_group[:, :, 0].view(batch_size, group_size, 1, 1)
                    v_grid = pts_group[:, :, 1].view(batch_size, group_size, 1, 1)
                    z_feat = pts_group[:, :, 2].view(batch_size, 1, group_size, 1)
                    z_feat = z_feat * (512 // 2) / 200
                    grid_2d = torch.cat([h_grid, v_grid], dim=-1)
                    del h_grid, v_grid, pts_group
                    pt_feat_2D_group = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=True,
                                               mode='bilinear', padding_mode='border')
                    del grid_2d

                    pt_feat_group = torch.cat([pt_feat_2D_group, z_feat], dim=1)
                    pt_output = self.mlp(pt_feat_group)  # shape = [batch_size, channels, point_num, 1]
                    pt_output = pt_output.permute([0, 2, 3, 1])
                    pt_output = pt_output.view([batch_size, group_size])
                    pt_sdf.append(pt_output.detach())

                    del pt_output,pt_feat_group,pt_feat_2D_group, z_feat
                pt_sdf = torch.cat(pt_sdf, dim=1)
                pt_sdf_list.append(pt_sdf)
                return pt_sdf_list

    def get_img_feature(self, img, no_grad=True):
        if no_grad:
            with torch.no_grad():
                f = self.hg(img)[-1]
            return f
        else:
            return self.hg(img)[-1]

    def get_vol_feature(self, vol, no_grad=True):
        if no_grad:
            with torch.no_grad():
                f = self.ve(vol, intermediate_output=False)
            return f
        else:
            return self.ve(vol, intermediate_output=False)