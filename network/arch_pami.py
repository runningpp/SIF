from __future__ import print_function, division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import network.hrnet as HRNET
import network.ve2 as ve2
import logging
import config
import time
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

class PamirNet(BaseNetwork):
    """PIVOIF implementation with multi-stage output"""

    def __init__(self):
        super(PamirNet, self).__init__()
        # self.hg = hg.HourglassNet(4, 4, 128, 64)
        # self.mlp = MLP()
        self.feat_ch_2D = 256
        self.feat_ch_3D = 32
        self.add_module('hrnet', HRNET.HRNetV2_W18_small_v2(
            inputMode='rgb_only', numOutput=self.feat_ch_2D, normLayer=nn.BatchNorm2d))
        self.add_module('ve', ve2.VolumeEncoder(3, self.feat_ch_3D))
        self.add_module('mlp', MLP(self.feat_ch_2D + self.feat_ch_3D, 1, weight_norm=False))

        logging.info('#trainable params of hourglass = %d' %
                     sum(p.numel() for p in self.hrnet.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))

    def forward(self, img, vol, pts, pts_proj):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        pt_sdf_list = []
        if self.training:
            img_feats = self.hrnet(img)
            vol_feats = self.ve(vol,intermediate_output=False)
            img_feats = img_feats[-len(vol_feats):]

            h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
            v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
            grid_2d = torch.cat([h_grid, v_grid], dim=-1)
            # pts *= 2.0  # corrects coordinates for torch in-network sampling
            x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
            y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
            z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
            grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)

            for img_feat, vol_feat in zip(img_feats, vol_feats):
                pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d,align_corners=False,
                                           mode='bilinear', padding_mode='border')
                pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d,align_corners=False,
                                           mode='bilinear', padding_mode='border')
                pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])
                pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
                pt_output = self.mlp(pt_feat)  # shape = [batch_size, channels, point_num, 1]
                pt_output = pt_output.permute([0, 2, 3, 1])
                pt_sdf = pt_output.view([batch_size, point_num])
                pt_sdf_list.append(pt_sdf)
            return pt_sdf_list
        else:
            with torch.no_grad():
                img_feats = self.hrnet(img)
                vol_feat = self.ve(vol, intermediate_output=False)[-1]
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
                    x_grid = pts_group[:, :, 0].view(batch_size, group_size, 1, 1, 1)
                    y_grid = pts_group[:, :, 1].view(batch_size, group_size, 1, 1, 1)
                    z_grid = pts_group[:, :, 2].view(batch_size, group_size, 1, 1, 1)
                    grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
                    del y_grid, x_grid,z_grid,pts_group
                    pt_feat_3D_group = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                               mode='bilinear', padding_mode='border')
                    pt_feat_3D_group = pt_feat_3D_group.view([batch_size, -1, group_size, 1])
                    del grid_3d

                    pts_group = pts_proj[:, start:end, :]
                    h_grid = pts_group[:, :, 0].view(batch_size, group_size, 1, 1)
                    v_grid = pts_group[:, :, 1].view(batch_size, group_size, 1, 1)
                    grid_2d = torch.cat([h_grid, v_grid], dim=-1)
                    del h_grid, v_grid, pts_group
                    pt_feat_2D_group = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                               mode='bilinear', padding_mode='border')
                    del grid_2d
                    # print(pt_feat_2D_group.shape,pt_feat_3D_group.shape)
                    pt_feat_group = torch.cat([pt_feat_2D_group, pt_feat_3D_group], dim=1)
                    pt_output = self.mlp(pt_feat_group)  # shape = [batch_size, channels, point_num, 1]
                    pt_output = pt_output.permute([0, 2, 3, 1])
                    pt_output = pt_output.view([batch_size, group_size])
                    pt_sdf.append(pt_output.detach())

                    del pt_output,pt_feat_group,pt_feat_2D_group, pt_feat_3D_group,
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

class PamirNet_v1(BaseNetwork):
    """PIVOIF implementation with multi-stage output"""

    def __init__(self):
        super(PamirNet_v1, self).__init__()
        # self.hg = hg.HourglassNet(4, 4, 128, 64)
        # self.mlp = MLP()
        self.feat_ch_2D = 64
        self.feat_ch_3D = 32
        self.add_module('hrnet', HRNET.HRNetV2_W18_small_v2(
            inputMode='rgb_only', numOutput=self.feat_ch_2D, normLayer=nn.BatchNorm2d))
        self.add_module('ve', ve2.VolumeEncoder1(3+1+3+7, self.feat_ch_3D))
        self.add_module('mlp', MLP(self.feat_ch_2D + self.feat_ch_3D, 1, weight_norm=False))

        logging.info('#trainable params of hourglass = %d' %
                     sum(p.numel() for p in self.hrnet.parameters() if p.requires_grad))
        logging.info('#trainable params of 3d encoder = %d' %
                     sum(p.numel() for p in self.ve.parameters() if p.requires_grad))
        logging.info('#trainable params of mlp = %d' %
                     sum(p.numel() for p in self.mlp.parameters() if p.requires_grad))

    def forward(self, img, vol, pts, pts_proj):
        """
        img: [batchsize * 3 (RGB) * img_h * img_w]
        pts: [batchsize * point_num * 3 (XYZ)]
        """
        batch_size = pts.size()[0]
        point_num = pts.size()[1]
        pt_sdf_list = []
        if self.training:
            t0 = time.time()
            vol_feats = self.ve(vol, intermediate_output=False)
            t00 = time.time()
            # print('vol feat',t00-t0)
            img_feats = self.hrnet(img)
            img_feats = img_feats[-len(vol_feats):]
            t1 = time.time()
            # print('img feats',t1-t00)
            h_grid = pts_proj[:, :, 0].view(batch_size, point_num, 1, 1)
            v_grid = pts_proj[:, :, 1].view(batch_size, point_num, 1, 1)
            grid_2d = torch.cat([h_grid, v_grid], dim=-1)
            # pts *= 2.0  # corrects coordinates for torch in-network sampling
            x_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1, 1)
            y_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1, 1)
            z_grid = pts[:, :, 2].view(batch_size, point_num, 1, 1, 1)
            grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
            t2 = time.time()
            # print('load time',t2-t1)
            for img_feat, vol_feat in zip(img_feats, vol_feats):
                pt_feat_2D = F.grid_sample(input=img_feat, grid=grid_2d,align_corners=False,
                                           mode='bilinear', padding_mode='border')
                t3 = time.time()
                pt_feat_3D = F.grid_sample(input=vol_feat, grid=grid_3d,align_corners=False,
                                           mode='bilinear', padding_mode='border')
                t4 = time.time()
                # print('2d feat',t3-t2,'3d feat',t4-t3)
                pt_feat_3D = pt_feat_3D.view([batch_size, -1, point_num, 1])
                pt_feat = torch.cat([pt_feat_2D, pt_feat_3D], dim=1)
                pt_output = self.mlp(pt_feat)  # shape = [batch_size, channels, point_num, 1]
                pt_output = pt_output.permute([0, 2, 3, 1])
                t5 = time.time()
                # print('mlp time',t5-t4)
                pt_sdf = pt_output.view([batch_size, point_num])
                pt_sdf_list.append(pt_sdf)
            return pt_sdf_list
        else:
            with torch.no_grad():
                img_feats = self.hrnet(img)
                vol_feat = self.ve(vol, intermediate_output=False)[-1]
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
                    x_grid = pts_group[:, :, 0].view(batch_size, group_size, 1, 1, 1)
                    y_grid = pts_group[:, :, 1].view(batch_size, group_size, 1, 1, 1)
                    z_grid = pts_group[:, :, 2].view(batch_size, group_size, 1, 1, 1)
                    grid_3d = torch.cat([x_grid, y_grid, z_grid], dim=-1)
                    del y_grid, x_grid,z_grid,pts_group
                    pt_feat_3D_group = F.grid_sample(input=vol_feat, grid=grid_3d, align_corners=False,
                                               mode='bilinear', padding_mode='border')
                    pt_feat_3D_group = pt_feat_3D_group.view([batch_size, -1, group_size, 1])
                    del grid_3d

                    pts_group = pts_proj[:, start:end, :]
                    h_grid = pts_group[:, :, 0].view(batch_size, group_size, 1, 1)
                    v_grid = pts_group[:, :, 1].view(batch_size, group_size, 1, 1)
                    grid_2d = torch.cat([h_grid, v_grid], dim=-1)
                    del h_grid, v_grid, pts_group
                    pt_feat_2D_group = F.grid_sample(input=img_feat, grid=grid_2d, align_corners=False,
                                               mode='bilinear', padding_mode='border')
                    del grid_2d
                    # print(pt_feat_2D_group.shape,pt_feat_3D_group.shape)
                    pt_feat_group = torch.cat([pt_feat_2D_group, pt_feat_3D_group], dim=1)
                    pt_output = self.mlp(pt_feat_group)  # shape = [batch_size, channels, point_num, 1]
                    pt_output = pt_output.permute([0, 2, 3, 1])
                    pt_output = pt_output.view([batch_size, group_size])
                    pt_sdf.append(pt_output.detach())

                    del pt_output,pt_feat_group,pt_feat_2D_group, pt_feat_3D_group,
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