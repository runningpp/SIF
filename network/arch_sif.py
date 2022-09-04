from __future__ import print_function, division, absolute_import
import os
import math
import numpy as np
import scipy
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import network.hg as hg
import network.hrnet as HRNET
import config

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
    def __init__(self, in_channels=257, out_channels=1, last_op = None,num_view=config.view_num):
        super(MLP, self).__init__()
        # inter_channels = (512, 256, 256, 128)
        inter_channels = (128, 128, 128, 128)

        self.num_view = num_view
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels[0],
                      kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inter_channels[0],
                      out_channels=inter_channels[1],
                      kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels[0],
                      out_channels=inter_channels[1],
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inter_channels[1],
                      out_channels=inter_channels[2],
                      kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels[1],
                      out_channels=inter_channels[2],
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=inter_channels[2],
                      out_channels=inter_channels[3],
                      kernel_size=1, stride=1, padding=0, bias=True),
            # nn.LeakyReLU(0.2, inplace=True)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels[2],
                      out_channels=inter_channels[3],
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=inter_channels[3], out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.last_op = last_op
        self.init_weights(init_type='normal', gain=0.02)

    def forward(self, x, weight):
        '''

        :param x: batch*view_num,dim(32+2+3+32+3),pts_num,1
        :param weight:
        :return:
        '''
        # print(x.shape)
        tmpx = x
        out = self.conv0(tmpx)
        # out = self.conv1(torch.cat([tmpx, out], dim=1))
        # out = self.conv2(torch.cat([tmpx, out], dim=1))
        out = out + self.conv1(out)
        out = out + self.conv2(out)  #batch*view,128,N,1

        if weight is not None:
        # merge multiview features
            if (config.view_num > 1):
                out = out.view(-1, config.view_num, out.shape[1], out.shape[2], out.shape[3])
                weight = weight.view(-1, config.view_num, weight.shape[1], weight.shape[2], weight.shape[3])

                # weight[:, 0, :, :, :] = torch.zeros_like(weight[:, 0, :, :, :])
                # weight[:, 1, :, :, :] = torch.zeros_like(weight[:, 1, :, :, :])
                # weight[:, 2, :, :, :] = torch.ones_like(weight[:, 2, :, :, :])

                out = out * weight
                out = out.sum(dim=1, keepdim=False)
        else:
            # if self.num_view> 1:
            #     if config.use_Attn:
            #         out = sdf_fusion(out,self.attn,view_num=2,label='encoder')
            out = out.view(-1, self.num_view, out.shape[1], out.shape[2], out.shape[3])
            if config.use_Attn:
                out = out.sum(dim=1, keepdim=False)
                # print(out.shape)
            else:
                out = out.mean(dim=1, keepdim=False)
        # out = self.conv3(torch.cat([tmpx, out], dim=1))
        out = out + self.conv3(out)
        out = self.conv4(out)
        if self.last_op is not None:
            out = self.last_op(out)
        return out


def cal_weight(dists,tau = 20,cutoff_dist = 500*0.00035):
    '''
    :param dists: (batch,pts_num,jts_num,1)
    :return:
    '''
    # compute cutoff weights

    v = tau * (dists - cutoff_dist)

    # v = v[...,None] #(batch,pts_num,dim,1)
    w = 1. - torch.sigmoid(v)#(batch,pts_num,jts_num,1)
    # print('weight',w)
    return w


class PIFUHrnet(BaseNetwork):
    def __init__(self):
        super(PIFUHrnet, self).__init__()

        self.add_module('hrnet', HRNET.HRNetV2_W18_small_v2(
            inputMode=config.input_mode, numOutput=32, normLayer=nn.BatchNorm2d))
	self.add_module('mlp', MLP(32 + 3 + 3 + 30, 1, nn.Sigmoid(), num_view=2))
        

    def get_feat_map(self, rgbd_imgs, masks):
        feat_map = self.hrnet(torch.cat([rgbd_imgs, masks], dim=1))[-1]
        return feat_map

    def forward(self, rgbd_imgs, masks,pts, pc_smpl_senmatics=None,pc_joints=None):
        """
        rgbd_imgs: [batchsize = batch_num * view_num, 7 (RGBDnormal), img_h, img_w]
        pts: [batchsize = batch_num * view_num, point_num, 3 (XYZ)]
        back_imgs:b,1+3,h,w
        pts_back:B,N,3
        pc_smpl_senmatics:batch,N,4
        pc_joints:batch*view_num,points_num,jts_num,1
        """
        batch_size = pts.size()[0]
        batch_num = batch_size // 2
        point_num = pts.size()[1]

        pt_sdf_list = []
        if pc_joints is not None:
            pc_joints = pc_joints.permute(0, 2, 1, 3)  # [batch_size,257,point_num,1]
            assert batch_num==pc_joints.shape[0]
            if config.use_cutoff:
                pc_joints = torch.clamp(pc_joints, 0, 0.175)
            joints_feat = torch.zeros(batch_size,30,point_num,1).cuda()
            joints_feat[0::2]=pc_joints
            joints_feat[1::2]=pc_joints

        if config.use_smpl_senmantic and pc_smpl_senmatics is not None:
            # print(pc_smpl_senmatics.shape)
            # if config.use_smpl_sdf:
            #     smpl_senmatics_feat = pc_smpl_senmatics[:, :, :1].permute(0, 2, 1).unsqueeze(-1)
            if config.only_use_smpl_sign:
                smpl_senmatics_feat = pc_smpl_senmatics[:,:,:1].permute(0, 2, 1).unsqueeze(-1)
            else:
                smpl_senmatics_feat = pc_smpl_senmatics.permute(0,2,1).unsqueeze(-1)  #B,4,N,1
            sdf_feats = torch.zeros(batch_size,smpl_senmatics_feat.shape[1],point_num,1).cuda()
            sdf_feats[0::2]=smpl_senmatics_feat
            sdf_feats[1::2]=smpl_senmatics_feat
            smpl_senmatics_feat = sdf_feats

        if self.training:
            h_grid = pts[:, :, 0].view(batch_size, point_num, 1, 1)  # normalized col indexes
            v_grid = pts[:, :, 1].view(batch_size, point_num, 1, 1)  # normalized row indexes
            grid = torch.cat([h_grid, v_grid], dim=-1)

            pt_mask = F.grid_sample(input=masks, grid=grid, mode='nearest', padding_mode='border', align_corners=True)
            # pt_mask = pt_mask.view(batch_size, 1, point_num, 1)
            del h_grid,v_grid
            # print('{} {}'.format(pt_depth.shape, pt_depth.dtype))
            pt_z = pts[:, :, -1].view(batch_size, 1, point_num, 1)
            if config.input_mode != 'normal':
                pt_depth = F.grid_sample(input=rgbd_imgs[:, 0, :, :].unsqueeze(1), grid=grid, mode='bilinear',
                                         padding_mode='border',
                                         align_corners=True)
                pt_depth = pt_depth.view(batch_size, 1, point_num, 1)

                pt_psdf = pt_z - pt_depth
                pt_psdf[0::2] = torch.clamp(pt_psdf[0::2], -config.psdf_zero_band_width, config.psdf_zero_band_width)
                pt_psdf[1::2] = torch.clamp(pt_psdf[1::2], - config.psdf_zero_band_width*5, config.psdf_zero_band_width*5)
            # pt_xyz = pts.clone().permute(0, 2, 1).unsqueeze(-1)
            # hard truncate
            # print(pt_psdf[0::2].reshape(-1))
            # print(pt_psdf[1::2].reshape(-1))
            # pt_mask = pt_mask.view(batch_size, 1, point_num, 1)

            img_feats = self.hrnet(rgbd_imgs)

            pt_normal= F.grid_sample(input=rgbd_imgs[:,1:], grid=grid, mode='bilinear',
                                     padding_mode='border',
                                     align_corners=True)
            ###########################
            # self.feat_map = img_feats[-1]
            for img_feat in img_feats:
                pt_feat = F.grid_sample(input=img_feat, grid=grid, mode='bilinear', padding_mode='border',
                                        align_corners=True)
                # print(pt_feat.shape) # [batch_size,256,point_num,1]
                # pt_feat = torch.cat([pt_feat, pt_z], dim=1)
                # print(pt_feat.shape) # [batch_size,257,point_num,1]

                # pt_feat = torch.cat([pt_feat, pt_psdf, pt_mask, pt_z], dim=1)
                # pt_feat = torch.cat([pt_feat, pt_psdf, pt_mask, pt_xyz], dim=1)

                del grid
                if config.input_mode == 'normal':
                    pt_feat = torch.cat([pt_feat, pt_mask, pt_normal, pt_z], dim=1)  # [batch_size*2,dim,N,1]
                else:
                    pt_feat = torch.cat([pt_feat, pt_psdf,pt_mask,pt_normal ,pt_z], dim=1)#[batch_size*2,dim,N,1]
                if config.use_joints_sample:
                    pt_feat = torch.cat([pt_feat, joints_feat], dim=1)
                if config.use_smpl_senmantic:
                    pt_feat = torch.cat([pt_feat,smpl_senmatics_feat],dim=1)
                    # print(pt_feat.shape)
                # print(pt_feat.shape)

                # print(pt_feat.shape)
                pt_sdf = self.mlp(pt_feat, None)
                # print(pt_sdf.shape) # [batch_num,1,point_num,1]
                pt_sdf = pt_sdf.view(batch_num, point_num)
                # print(pt_sdf.shape) # [batch_num,point_num]
                pt_sdf_list.append(pt_sdf)
        else:
            with torch.no_grad():

                img_feats = self.hrnet(rgbd_imgs)

                self.feat_map = img_feats[-1]

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
                    # joints_feat_group = joints_feat[:,:,start:end,...]
                    index_invalid = (pts_group[:, :, 0] >= 1) + (pts_group[:, :, 0] <= -1) +\
                                    (pts_group[:, :,1] >= 1) + (pts_group[:, :,1] <= -1)
                    pts_group[index_invalid] = -1
                    # index_invalid = index_invalid.view(batch_size, 1, -1)
                    # print(index_invalid.sum(),index_invalid.shape)
                    if config.use_smpl_senmantic:
                        smpl_feat_group = smpl_senmatics_feat[:, :, start:end, ...]
                    if config.use_joints_sample:
                        joints_feat_group = joints_feat[:,:,start:end,...]
                    # print(pts_group.shape) # [batch_size, group_size, 3]
                    h_grid = pts_group[:, :, 0].view(batch_size, group_size, 1, 1)
                    v_grid = pts_group[:, :, 1].view(batch_size, group_size, 1, 1)
                    grid = torch.cat([h_grid, v_grid], dim=-1)
                    del h_grid,v_grid
                    pt_group_feat = F.grid_sample(input=img_feats[-1], grid=grid, mode='bilinear',
                                                  padding_mode='border', align_corners=True)
                    # print(pt_group_feat.shape) # [batch_size,256,group_size,1]

                    pt_group_normal = F.grid_sample(input=rgbd_imgs[:, 1:], grid=grid, mode='bilinear',
                                              padding_mode='border',
                                              align_corners=True)
                    # print(rgbd_imgs.shape) # [batch_size, 4, 512, 512]
                    pt_group_z = pts_group[:, :, 2].view(batch_size, 1, group_size, 1)

                    if config.input_mode != 'normal':
                        pt_group_depth = F.grid_sample(input=rgbd_imgs[:, 0, :, :].unsqueeze(1), grid=grid,
                                                       mode='bilinear',
                                                       padding_mode='border', align_corners=True)
                        pt_group_depth = pt_group_depth.view(batch_size, 1, group_size, 1)

                        pt_group_psdf = pt_group_z - pt_group_depth
                        pt_group_psdf[0::2] = torch.clamp(
                            pt_group_psdf[0::2], -config.psdf_zero_band_width, config.psdf_zero_band_width)
                        pt_group_psdf[1::2] = torch.clamp(
                            pt_group_psdf[1::2], -config.psdf_zero_band_width*5, config.psdf_zero_band_width*5)
                    # pt_group_xyz = pts_group[:, :, 0:3].clone().permute(0, 2, 1).unsqueeze(-1)

                    # hard truncate
                    pt_group_mask = F.grid_sample(input=masks, grid=grid, mode='nearest', padding_mode='border',
                                                  align_corners=True)


                    #####################################

                    ############################
                    # if config.useRFF:
                    #     pt_group_feat = torch.cat([pt_group_feat, pt_group_psdf, pt_group_mask, pt_group_pos_feat],
                    #                               dim=1)
                    # else:
                        # pt_group_feat = torch.cat([pt_group_feat, pt_group_psdf, pt_group_mask, pt_group_z], dim=1)
                    if config.input_mode != 'normal':
                        pt_group_feat = torch.cat([pt_group_feat, pt_group_psdf, pt_group_mask,pt_group_normal, pt_group_z], dim=1)
                    else:
                        pt_group_feat = torch.cat(
                            [pt_group_feat, pt_group_mask, pt_group_normal, pt_group_z], dim=1)
                    # pt_group_feat = torch.cat([pt_group_feat, pt_group_z], dim=1)
                    # print(pt_group_feat.shape) # [batch_size,257,group_size,1]
                    if config.use_joints_sample:
                        pt_group_feat = torch.cat([pt_group_feat,joints_feat_group],dim=1)
                    if config.use_smpl_senmantic:
                        pt_group_feat = torch.cat([pt_group_feat,smpl_feat_group],dim=1)
                        del smpl_feat_group
                    pt_group_sdf = self.mlp(pt_group_feat, None)
                    # print(pt_group_sdf.shape) # [batch_num,1,group_size,1]
                    pt_group_sdf = pt_group_sdf.view(batch_num, group_size)

                    pt_sdf.append(pt_group_sdf.detach())
                    del pt_group_sdf,pt_group_feat,pts_group, pt_group_mask, pt_group_z,
                    if config.input_mode != 'normal':
                        del pt_group_psdf
                pt_sdf = torch.cat(pt_sdf, dim=1)
                pt_sdf_list.append(pt_sdf)
                del img_feats,pt_sdf
            # print(len(pt_sdf_list)) # 1, for inference, we only use the last feature map of hourglass net
            # print(pt_sdf_list[0].shape) # [batch_num, group_size]
        return pt_sdf_list


