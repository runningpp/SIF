"""Data loader"""

from __future__ import division, print_function

import os
import glob
import numpy as np
import scipy.io as sio
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import math
from torchvision import transforms, utils
import config
import time
import trimesh
import trimesh.proximity
from colorsys import hsv_to_rgb
import pickle as pkl
from psbody.mesh import Mesh
from scipy.spatial import cKDTree as KDTree
import torch.nn.functional as F
from PIL import Image

torch.utils.backcompat.broadcast_warning.enabled = True
def save_point_cloud_as_obj(pc, path):
    with open(path, 'w') as f:
        for i in range(pc.shape[0]):
            f.write('v %f %f %f\n' % (pc[i, 0], pc[i, 1], pc[i, 2]))
def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)

def get_boundary_pixels(depth, win_size=3, thres=0.05):
    """Returns a mask indicating pixels that are on the depth boundaries """
    depth_rolled = []
    r_roll_steps = np.array(range(win_size)) - (win_size // 2)
    c_roll_steps = np.array(range(win_size)) - (win_size // 2)
    for r_roll_step in r_roll_steps:
        for c_roll_step in c_roll_steps:
            if r_roll_step == 0 and c_roll_step == 0:
                continue
            depth_ = np.roll(depth, r_roll_step, axis=0)
            depth_ = np.roll(depth_, c_roll_step, axis=1)
            depth_rolled.append(depth_)
    depth_rolled = np.asarray(depth_rolled)
    diff = np.abs(depth_rolled - np.expand_dims(depth, axis=0))
    max_diff = np.max(diff, axis=0)
    m = np.uint8(max_diff > thres)
    return m
def get_smpl30_skeleton():
    return np.array([
        [0,1],[1,2],[2,6],
        [6,7],[7,8],[8,9],
        [6,10],[10,11],[11,12],
        [6,3],[3,4],#[4,5],
        #[5,20],[20,22],
        #[5,21],[21,23],
        [0,13],
        [13,14],[14,15],[15,16],[16,29],
        #[16,27],[27,28],
        [13,17],[17,18],[18,19],[19,26],
        #[19,24],[24,25]
    ])

class ImgDataset(Dataset):
    def __init__(self, dataset_dir, img_h, img_w, training, train_color, testing_res):
        super(ImgDataset, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.training = training
        self.train_color = train_color
        self.dataset_dir = dataset_dir
        self.subject_number = config.subject_number
        self.render_view_num = config.render_view_number
        self.testing_res = testing_res

        self.root = self.dataset_dir
        self.RENDER = os.path.join(self.root, 'img')
        self.PARAM = os.path.join(self.root, 'parameter')
        self.NORMAL = os.path.join(self.root, 'normal')
        self.MASK = os.path.join(self.root, 'mask')
        self.DEPTH = os.path.join(self.root, 'depth')
        self.SMPL = os.path.join(self.root, 'smpl')
        self.smpl_template_v = self.get_smpl_template_v()
        self.OBJ = '/home/lpp/data/TwinDom_PerspectRandom_Noisy'
        self.smpl_pth = '/home/lpp/data/Twindom_smpl_915'
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if training:
            self.interval = 6
        else:
            self.interval = 6*3

        print(dataset_dir)

        # print(dataset_dir)
        if training:
            names = sorted(os.listdir(self.smpl_pth))

            data_list = []

            names = names[:config.subject_number]

            for j in names:

                for i in range(0, 360,6):
                    data_list.append(os.path.join(self.MASK,j,'%d.png'%i))

            self.data_list = data_list

            print(len(self.data_list))
        else:
            names = sorted(os.listdir(self.smpl_pth))[:300]
            data_list = []
            for j in names:
                for i in range(0, 30*6, 3*6):
                    data_list.append(os.path.join(self.MASK, j, '%d.png' % i))

            self.data_list = data_list

            print(len(self.data_list))

    def get_smpl_template_v(self):

        # smpl_pth = '/home/lpp/Projects/EasyMocap/data/smplx/smpl/SMPL_NEUTRAL.pkl'
        # with open(smpl_pth, 'rb') as smpl_file:
        #     data = pkl.load(smpl_file, encoding='latin1')
        # v_template = data['v_template']
        # print(v_template.shape)
        #
        # v_min = v_template.min(axis=0)
        # v_max = v_template.max(axis=0)
        # center = (v_min + v_max) / 2
        # height = (v_max - v_min).max()
        # v_template = (v_template - center) / height
        v_template = torch.load('./dataloader/smpl_v_template.pkl')
        return v_template
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        global_img_id = item

        self.cam_f = config.cam_f

        center_depth_perturbation = np.random.randn(self.render_view_num) * 0.02

        if self.training:
            subject_id = item // int(self.render_view_num)
            vid = item % int(self.render_view_num) * self.interval

            point_num = 8000
            # sigma = 0.025
            sigma = 0.05

            data_fd, name = os.path.split(self.data_list[global_img_id])
            subject_name = os.path.split(data_fd)[-1]
            # print(subject_name,data_fd,name)
            assert vid ==  int(name[:-4])

            color,normal,mask,extrinsic,intrinsic = self.get_render(subject_name,vid)

            smpl_vis_depth,center_smpl = self.load_vis_smpl_depth(vid,subject_name)
            center_smpl += center_depth_perturbation[vid // self.interval]


            pc_pts, pts_list, pts_label,pts_smpl_senmatics = self.sample_points_multiview(subject_name, vid, center_smpl,
                                                                                          extrinsic,intrinsic,
                                                                                          point_num=point_num,sigma=sigma)



            smpl_b_depth,smpl_b_center, smpl_intrinsic, smpl_extrinsic = self.load_b_smpl_depth(vid, subject=subject_name)
            back_normal_img,back_mask = self.load_back_normal(vid,subject=subject_name)
            # print(back_mask.shape,'back mask')
            trans_pc = np.matmul(pc_pts,smpl_extrinsic[:3,:3].T)+smpl_extrinsic[:,3][None,:]
            # print(pc_pts.shape,trans_pc.shape)
            # cx = smpl_intrinsic[0, 2]
            # cy = smpl_intrinsic[1, 2]
            # fx = smpl_intrinsic[0, 0]
            # fy = smpl_intrinsic[1, 1]
            trans_pc = torch.FloatTensor(trans_pc)  # pts_num,3
            trans_pc[:, 0] = (trans_pc[:, 0] * smpl_intrinsic[0, 0] / trans_pc[:, 2] + smpl_intrinsic[0, 2]
                              -config.img_w/2) / (config.img_w/2)
            trans_pc[:, 1] = (trans_pc[:, 1] * smpl_intrinsic[1, 1] / trans_pc[:, 2] + smpl_intrinsic[
                1, 2]-config.img_h/2) / (config.img_h/2)
            trans_pc[:,2]-=smpl_b_center



            jts_pth = os.path.join(self.smpl_pth,subject_name,'joints30_scaled.pkl')
            joints = torch.as_tensor(torch.load(jts_pth),dtype=torch.float32) #30,3
            #
            pc_pts = torch.FloatTensor(pc_pts) #pts_num,3
            pc_joints = pc_pts[:,None,:]-joints[None,...]
            # print(pc_joints.shape)
            if config.use_smpl_senmantic:
                return {
                    'subject_id': subject_id,
                    # 'rgbd_imgs': torch.stack(rgbd_img_list, dim=0),
                    # 'depth_masks_dt': torch.stack(depth_mask_dt_list, dim=0),
                    # 'pts': torch.stack(pts_list, dim=0),
                    'pts_label': torch.FloatTensor(pts_label),
                    'joints': pc_pts,
                    'pc_joints': pc_joints,  # N,jts_num,3
                    # 'pc_smpl_depth_sdf': pt_smpl_psdf,  # N
                    # 'back_depth_normal': torch.cat([smpl_depth, back_normal_img], dim=0),
                    'depth_normal': torch.stack([torch.cat([smpl_vis_depth, normal]), torch.cat([smpl_b_depth, back_normal_img])], dim=0),
                    # 2,4,512,512
                    'mask': torch.stack([mask, back_mask], dim=0),  # 2,1,512,512
                    # 'pts_back':trans_pc,  # N,3
                    'pts': torch.stack([pts_list[0], trans_pc], dim=0),  # 2,N,3
                    'pc_smpl_semantic': pts_smpl_senmatics,  # N,4
                    'color': color,
                }
            return {
                'subject_id': subject_id,
                # 'rgbd_imgs': torch.stack(rgbd_img_list, dim=0),
                # 'depth_masks_dt': torch.stack(depth_mask_dt_list, dim=0),
                # 'pts': torch.stack(pts_list, dim=0),
                'pts_label': torch.FloatTensor(pts_label),
                'joints': pc_pts,
                'pc_joints': pc_joints,  # N,jts_num,3
                # 'pc_smpl_depth_sdf': pt_smpl_psdf,  # N
                # 'back_depth_normal': torch.cat([smpl_depth, back_normal_img], dim=0),
                'depth_normal': torch.stack(
                    [torch.cat([smpl_vis_depth, normal]), torch.cat([smpl_b_depth, back_normal_img])], dim=0),
                # 2,4,512,512
                'mask': torch.stack([mask, back_mask], dim=0),  # 2,1,512,512
                # 'pts_back':trans_pc,  # N,3
                'pts': torch.stack([pts_list[0], trans_pc], dim=0),  # 2,N,3
                # 'pc_smpl_semantic': pts_smpl_senmatics,  # N,4
                'color': color,
            }

        else:
            # load multi-view rgbd images

            data_fd, name = os.path.split(self.data_list[global_img_id])
            subject_name = os.path.split(data_fd)[-1]
            vid = int(name[:-4])
            self.view = vid
            # assert view_id * 3 == vid,print(vid,view_id)

            # print(vid)
            view_ids = [vid]

            ######################################################

            print('get item: %d' % item)
            color, normal, mask, extrinsic, intrinsic = self.get_render(subject_name, vid)
            smpl_f_depth, center_smpl = self.load_vis_smpl_depth(vid, subject=subject_name)

            tt0 = time.time()
            x_coords = torch.linspace(-0.5, 0.5, steps=self.testing_res, device=config.device).float().detach()
            y_coords = x_coords.clone()
            z_coords = x_coords.clone()
            xv, yv, zv = torch.meshgrid(x_coords, y_coords, z_coords)  # (256, 256, 256)
            xv = torch.reshape(xv, (-1, 1))  # (256*256*256, 1)
            yv = torch.reshape(yv, (-1, 1))
            zv = torch.reshape(zv, (-1, 1))
            pts = torch.cat([xv, yv, zv], dim=-1)  # (256*256*256, 3)
            pts_flag = torch.ones(pts.shape[0], dtype=torch.float, device=config.device)

            # # print(grid_coords.shape)
            # grid_coords = pts.clone()
            # grid_coords[:, 0], grid_coords[:, 2] = pts[:, 2], pts[:, 0]
            #
            # grid_coords = 2 * grid_coords  # / scale
            # grid_coords = torch.tensor(grid_coords, dtype=torch.float32)




            extrinsic, intrinsic = torch.FloatTensor(extrinsic).to(config.device),torch.FloatTensor(intrinsic).to(config.device)
            pts_ = (torch.matmul(pts.unsqueeze(-2), extrinsic[:3,:3].t()) + extrinsic[:3,3]).detach().clone()
            pts_ = pts_.squeeze(-2)

            pts_[:, 0] = (pts_[:, 0] * intrinsic[0, 0] / pts_[:, 2] + intrinsic[0, 2]
                              - config.img_w / 2) / (config.img_w / 2)
            pts_[:, 1] = (pts_[:, 1] * intrinsic[1, 1] / pts_[:, 2] + intrinsic[
                1, 2] - config.img_h / 2) / (config.img_h / 2)
            pts_[:, 2] -= center_smpl


            pts_flag[pts_flag < config.view_num - 1e-6] = 0

            pts = pts[pts_flag > config.view_num - 1e-6, ...]# pts_num,3

            if config.use_smpl_depth:
                vid = view_ids[0]
                assert len(view_ids) == 1

                smpl_depth, smpl_center, smpl_intrinsic, smpl_extrinsic = self.load_b_smpl_depth(vid,
                                                                                               subject=subject_name)
                back_normal_img,back_mask = self.load_back_normal(vid, subject=subject_name)

                smpl_extrinsic = torch.FloatTensor(smpl_extrinsic).to(config.device)
                # print(pts.shape,smpl_extrinsic.shape)
                trans_pc = torch.matmul(pts, smpl_extrinsic[:3, :3].t()) + smpl_extrinsic[:, 3].unsqueeze(0)

                # trans_pc = torch.FloatTensor(trans_pc)  # pts_num,3
                trans_pc[:, 0] = (trans_pc[:, 0] * smpl_intrinsic[0, 0] / trans_pc[:, 2] + smpl_intrinsic[0, 2]
                                  - config.img_w / 2) / (config.img_w / 2)
                trans_pc[:, 1] = (trans_pc[:, 1] * smpl_intrinsic[1, 1] / trans_pc[:, 2] + smpl_intrinsic[
                    1, 2] - config.img_h / 2) / (config.img_h / 2)

                trans_pc[:, 2] -= smpl_center



            jts_pth = os.path.join(self.smpl_pth, subject_name, 'joints30_scaled.pkl')
            joints = torch.as_tensor(torch.load(jts_pth), dtype=torch.float32).to(config.device)  # 30,3
            pc_joints = pts[:, None, :] - joints[None, ...]



            if config.use_smpl_senmantic:
                smpl = os.path.join(self.smpl_pth, subject_name, 'smpl_mesh_scaled.obj')
                mesh = trimesh.load(smpl)
                pts_numpy = pts.cpu().numpy()
                smpl_m = Mesh(filename=smpl)
                smpl_v = smpl_m.v
                kdtree = KDTree(smpl_v)
                pts_dst, pts_idx = kdtree.query(pts_numpy)
                pts_smpl_v = self.smpl_template_v[pts_idx]
                if config.use_smpl_sdf:
                    t0 = time.time()
                    print(pts_numpy.shape)
                    n = os.path.split(data_fd)[-1]
                    f_n = os.path.join('/home/user/lpp/projects/rgbd_v/sdf_smpl',
                                       n + '_view%d.pkl' % view_ids[0])
                    if os.path.exists(f_n):
                        pts_dst = torch.load(f_n)
                    else:
                        assert os.path.exists(f_n)
                        # if pts_numpy.shape[0] < 2800000:
                        #     pts_dst = mul_cal_sdf(mesh, pts_dst, pts_numpy, batch_points=20000, num_cores=10)
                        # else:
                        #     pts_dst = mul_cal_sdf(mesh, pts_dst, pts_numpy, batch_points=20000, num_cores=10)
                        # print('end cal sdf', time.time() - t0)
                        # torch.save(pts_dst, f_n)
                # else:
                #     flag = mesh.contains(pts_numpy)
                #     sign = np.ones(pts_numpy.shape[0])
                #     sign[~flag] = -1
                #     pts_dst = pts_dst * sign

                pts_dst = np.clip(pts_dst, a_max=0.077, a_min=-0.25)[:, None]

                pts_smpl = torch.FloatTensor(np.concatenate([pts_dst, pts_smpl_v], axis=1)).to(device=config.device)
                return{
                'subject_id': subject_name,
                'pc_smpl_semantic': pts_smpl,
                # 'rgbd_imgs': rgbd_img_list,
                'depth_normal': torch.stack([torch.cat([smpl_f_depth,normal]),
                                             torch.cat([smpl_depth, back_normal_img])], dim=0).to(config.device),
                # 'depth_masks_dt': depth_mask_dt_list,
                'mask': torch.stack([mask, back_mask], dim=0).to(config.device),
                # 'pts': pts_list,
                'pts': torch.stack([pts_, trans_pc], dim=0),
                'pts_flag': pts_flag,
                # 'cam_r_list': cam_r_list,
                # 'cam_t_list': cam_t_list,
                # 'center_depth_list': center_depth_list,
                'pc_joints': pc_joints,
                # 'pc_smpl_depth_sdf': pt_smpl_psdf,  # N
                # 'back_depth_normal': torch.cat([smpl_depth, back_normal_img],dim=0).to(config.device),
                # 'pts_back': trans_pc,  # N,3

            }

            return {
                'subject_id': subject_name,
                # 'rgbd_imgs': rgbd_img_list,
                'depth_normal': torch.stack([torch.cat([smpl_f_depth, normal]),
                                             torch.cat([smpl_depth, back_normal_img])], dim=0).to(config.device),
                # 'depth_masks_dt': depth_mask_dt_list,
                'mask': torch.stack([mask, back_mask], dim=0).to(config.device),
                # 'pts': pts_list,
                'pts': torch.stack([pts_, trans_pc], dim=0),
                'pts_flag': pts_flag,
                # 'cam_r_list': cam_r_list,
                # 'cam_t_list': cam_t_list,
                # 'center_depth_list': center_depth_list,
                'pc_joints': pc_joints,
                # 'pc_smpl_depth_sdf': pt_smpl_psdf,  # N
                # 'back_depth_normal': torch.cat([smpl_depth, back_normal_img],dim=0).to(config.device),
                # 'pts_back': trans_pc,  # N,3

            }

    def get_render(self, subject, vid):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''


        # The ids are an even distribution of num_views around view_id
        # print(view_ids)

        extrinsic_path = os.path.join(self.PARAM, subject, '{}_extrinsic.npy'.format(vid))
        intrinsic_path = os.path.join(self.PARAM, subject, '{}_intrinsic.npy'.format(vid))
        render_path = os.path.join(self.RENDER, subject, '{}.png'.format(vid))
        # depth_path = os.path.join(self.DEPTH, subject, '{}.npz'.format(vid))
        # if os.path.exists(depth_path):
        #     depth = Image.fromarray(np.load(depth_path)['arr_0'])
        # else:
        #     print('warning: no depth')
        #     assert 'no depth'

        normal_path = os.path.join(self.NORMAL, subject, '{}.png'.format(vid))
        # normal_b_path = os.path.join(self.NORMAL, subject, '{}.png'.format((vid+180)%360))
        mask_path = os.path.join(self.MASK, subject, '{}.png'.format(vid))
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.MASK, subject, '{}.jpg'.format(vid))

        # loading calibration data
        extrinsic = np.load(extrinsic_path)
        intrinsic = np.load(intrinsic_path)

        if os.path.exists(render_path):
            render = Image.open(render_path).convert('RGB')
        else:
            print('warning: no render')

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('RGB')  # 512,512,3
        else:
            print('warning: no mask')

        if os.path.exists(normal_path):
            normal = Image.open(normal_path).convert('RGB')
        else:
            print('warning: no render')


        # if os.path.exists(normal_b_path):
        #     normal_b = Image.open(normal_b_path)
        # else:
        #     print('warning: no render back')
        #     normal_b = render.copy()

        S = np.array(mask).shape[0]


        intrinsic[1, :] *= -1.0
        intrinsic[1, 2] += S


        # calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        # # rot = extrinsic[:3, :3].T
        # # trans = - rot @ extrinsic[:3, 3]
        # # c2w = torch.cat((torch.Tensor(rot), torch.Tensor(trans).reshape(3, 1)), dim=1)
        # # extrinsic = torch.Tensor(extrinsic[:, :3]).float()
        # extrinsic = torch.Tensor(extrinsic).float()

        mask = torch.FloatTensor(np.array(mask))[:, :, 0] / 255
        mask = mask.reshape(1, S, S)
        render = self.to_tensor(render) * mask +(1-mask) # white background

        normal = self.to_tensor(normal) * mask + (1-mask)



        # depth = np.array(depth)
        # if len(depth.shape) >= 3:
        #     depth_list.append((torch.FloatTensor(depth[:, :, 0]) * mask).unsqueeze(0))
        # else:
        #     depth_list.append((torch.FloatTensor(depth) * mask).unsqueeze(0))
        return render,normal,mask,extrinsic,intrinsic

    def sample_points_multiview(self, subject_name, vid, center_smpl,extrinsic,intrinsic,
                                point_num=8000, sigma=0.05):

        # mesh = self.mesh_dic[subject]
        if os.path.exists(os.path.join(self.OBJ,subject_name, 'sample_points.npz' )):
            # print('load from npz')
            file = os.path.join(self.OBJ,subject_name, 'sample_points.npz')
            boundary_samples_npz = np.load(file)
            inside = boundary_samples_npz['pts_inside']
            outside = boundary_samples_npz['pts_outside']

            if config.use_smpl_senmantic:
                if os.path.exists(os.path.join(self.OBJ,subject_name, 'sample_points_smpl_semantic.npz')):
                    pth = os.path.join(self.OBJ,subject_name, 'sample_points_smpl_semantic.npz')
                    data = np.load(pth)
                    # smpl_template_points = template_smpl, sign_dist = sign_dist
                    template_smpl = data['smpl_template_points']  # N,3
                    if config.use_smpl_sdf:
                        pth = os.path.join(self.OBJ,subject_name, 'sample_points_sdf.npz')
                        data = np.load(pth)
                        sign_dist = data['sign_dist']
                        sign_dist = np.clip(sign_dist, a_min=-0.25, a_max=0.077)[:, None]  # N,1
                        # print('use sdf smpl')
                    else:
                        sign_dist = data['sign_dist']
                        sign_dist = np.clip(sign_dist, a_min=-0.25, a_max=0.077)[:,None] #N,1
                    # print(sign_dist.shape)
                    num_inside = inside.shape[0]
                    inside = np.concatenate([inside,sign_dist[:num_inside],template_smpl[:num_inside]],axis=1)
                    outside = np.concatenate([outside,sign_dist[num_inside:],template_smpl[num_inside:]],axis=1)
                    # print(inside.shape,outside.shape)

            sample_nums = point_num
            np.random.shuffle(inside)
            np.random.shuffle(outside)
            nin = inside.shape[0]
            inside_points = inside[
                            :sample_nums // 2] if nin > sample_nums // 2 else inside
            outside_points = outside[
                             :sample_nums // 2] if nin > sample_nums // 2 else outside[:(sample_nums - nin)]
        else:
            # print('lpppp')
            try:
                mesh = trimesh.load(os.path.join(self.OBJ,subject_name, '%s_scaled_meshOnly.ply' % subject_name[:-2]),
                                    use_embree=True)
            except:
                mesh = trimesh.load(os.path.join(self.OBJ,subject_name, '%s_scaled.ply' % subject_name[:-2]),
                                    use_embree=True)

            surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * point_num)


            curv_radius = 0.002
            curv_thresh = 0.004
            curvs = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, surface_points, curv_radius)
            curvs = abs(curvs)
            curvs = curvs / max(curvs)  # normalize curvature

            sigmas = np.zeros(curvs.shape)
            sigmas[curvs <= curv_thresh] = sigma
            sigmas[curvs > curv_thresh] = sigma / 5
            random_shifts = np.random.randn(surface_points.shape[0], surface_points.shape[1])
            random_shifts[:, 0] *= sigmas
            random_shifts[:, 1] *= sigmas
            random_shifts[:, 2] *= sigmas
            surface_points = surface_points + random_shifts

            volume_pnum = point_num // 4
            volume_points = (np.random.random((volume_pnum, 3)) * (np.ones((3,)) * 1.0)) + (np.ones((3,)) * (-0.5))

            points = np.float32(np.concatenate([surface_points, volume_points], axis=0))
            np.random.shuffle(points)

            # [NOTE]
            # draw uniform samples in a small ball, which are then use to perturb the points before
            # determining whether or not they are inside the mesh
            # this is an ugly trick to approximate the SDF calculation
            uvw = np.random.randn(*points.shape)
            uvw = uvw / np.linalg.norm(uvw, axis=-1, keepdims=True)
            r_3 = np.random.rand(*(points.shape[:-1] + (1,)))
            r = r_3 ** (1.0 / 3.0) * 0.005
            perturbation = r * uvw

            inside = mesh.contains(points + perturbation)
            inside_points = points[inside]
            outside_points = points[np.logical_not(inside)]

            inside_pnum = inside_points.shape[0]

            inside_points = inside_points[
                            :point_num // 2] if inside_pnum > point_num // 2 else inside_points

            outside_points = outside_points[
                             :point_num // 2] if inside_pnum > point_num // 2 else outside_points[
                                                                                   :(point_num - inside_pnum)]

        jts_pth = os.path.join(self.smpl_pth, subject_name, 'joints30_scaled.pkl')
        joints = torch.as_tensor(torch.load(jts_pth), dtype=torch.float32)  # 30,3
        if config.use_joints_sample:
            bones = get_smpl30_skeleton()
            I = bones[:, 0]
            J = bones[:, 1]
            start_joints = joints[I]
            end_joints = joints[J]
            bones_list = []
            for i in range(0, 11):
                i = i / 10.
                jts = i * start_joints + (1 - i) * end_joints + torch.randn(start_joints.shape) * 0.005
                bones_list.append(jts)
            bones_list = np.concatenate(bones_list, axis=0)
            if config.use_smpl_senmantic:
                smpl = os.path.join(self.smpl_pth, subject_name, 'smpl_mesh_scaled.obj')
                smpl_m = Mesh(filename=smpl)
                smpl_v = smpl_m.v
                kdtree = KDTree(smpl_v)
                joints_dst, joints_idx = kdtree.query(bones_list)
                if config.use_smpl_sdf:
                    mesh = trimesh.load(smpl)
                    joints_dst = trimesh.proximity.signed_distance(mesh,bones_list)
                joints_dst = np.clip(joints_dst,a_max=0.077,a_min=0)[:,None]
                joints_smpl_v = self.smpl_template_v[joints_idx]
                bones_list = np.concatenate([bones_list,joints_dst,joints_smpl_v],axis=1)
            # print(bones_list.shape)

            inside_points = np.concatenate([inside_points,bones_list],axis=0)
            # print('use joints sample',inside_points.shape)

        if config.use_smpl_senmantic:
            pts_ = np.concatenate([inside_points, outside_points], axis=0)
            pts = pts_[:,:3]
            pts_smpl_semantics = pts_[:,3:]
            pts_smpl_semantics= torch.FloatTensor(pts_smpl_semantics)
        else:
            pts = np.concatenate([inside_points, outside_points], axis=0)
        pts_ov = np.float32(
            np.concatenate([np.ones((inside_points.shape[0])), np.zeros((outside_points.shape[0]))], axis=0))

        pc_pts = pts.copy()
        # debug
        # if (not mesh.is_watertight):
        #     print('mesh of subject ' + subject_name[:-2] + ' is not watertight.')
        #     filename = ('./debug/sampling_points' + subject_name[:-2] + '.obj')
        #     with open(filename, 'w') as fp:
        #         for i in range(pts.shape[0]):
        #             point = pts[i,:]
        #             ov = pts_ov[i]
        #             if (ov > 0.5):
        #                 fp.write('v %f %f %f 0 1 0\n' % (point[0], point[1], point[2]))
        #             else:
        #                 fp.write('v %f %f %f 1 0 0\n' % (point[0], point[1], point[2]))
        # os._exit(0)

        pts_list = []
        pts_ov_list = []

        for view in [vid]:
            pts_copy = (np.matmul(pts, extrinsic[:3, :3].T) + extrinsic[:, 3][None, :]).copy()
            # pts_copy = (np.dot(pts, cam_R.transpose()) + np.expand_dims(cam_t, axis=0)).copy()

            # # debug
            # filename = ('sampling_points_view_%d_%s.obj' % (view,subject_name[:-2]))
            # with open(filename, 'w') as fp:
            #     for i in range(pts.shape[0]):
            #         point = pts[i,:]
            #         ov = pts_ov[i]
            #         if (ov > 0.5):
            #             fp.write('v %f %f %f 0 1 0\n' % (point[0], point[1], point[2]))
            #         else:
            #             fp.write('v %f %f %f 1 0 0\n' % (point[0], point[1], point[2]))
            # # os._exit(0)
            pts_copy[:, 0] = (pts_copy[:, 0] * intrinsic[0, 0] / pts_copy[:, 2] + intrinsic[0, 2]
                              - config.img_w / 2) / (config.img_w / 2)
            pts_copy[:, 1] = (pts_copy[:, 1] * intrinsic[1, 1] / pts_copy[:, 2] + intrinsic[1, 2]
                              - config.img_h / 2) / (config.img_h / 2)
            pts_copy[:, 2] -= center_smpl

            pts_list.append(torch.FloatTensor(pts_copy))
        if config.use_smpl_senmantic:
            return  pc_pts, pts_list, pts_ov,pts_smpl_semantics
        return pc_pts, pts_list, pts_ov,None

    def load_b_smpl_depth(self,vid,subject):
        assert vid < 361
        vid = (vid+180) % 360
        smpl_depth_pth = os.path.join(self.SMPL,subject,'smpl_depth_%d.npz'%vid)
        smpl_intrinsic = os.path.join(self.PARAM,subject,'%d_intrinsic.npy'%vid)
        smpl_extrinsic = os.path.join(self.PARAM,subject,'%d_extrinsic.npy'%vid)
        extrinsic = np.load(smpl_extrinsic)
        intrinsic = np.load(smpl_intrinsic)
        # print(smpl_depth_pth)

        intrinsic[1, :] *= -1.0
        intrinsic[1, 2] += 512


        if os.path.exists(smpl_depth_pth):
            depth =np.load(smpl_depth_pth)['arr_0']
            mask = depth<1e-3
            depth = 1./(depth+1e-6)
            center = np.mean(depth[~mask])
            depth -= center
            depth[mask] = 1.
            # depth[mask] = 10
        else:
            print('warning: no depth')
            assert 'no depth'

        return torch.FloatTensor(depth[None,...]),center,intrinsic,extrinsic

    def load_vis_smpl_depth(self,vid,subject):
        vid = vid
        smpl_depth_pth = os.path.join(self.SMPL,subject,'smpl_depth_%d.npz'%vid)
        # print(smpl_depth_pth)
        if os.path.exists(smpl_depth_pth):
            depth =np.load(smpl_depth_pth)['arr_0']
            mask = depth<1e-3
            depth = 1./(depth+1e-6)
            center = np.mean(depth[~mask])
            depth -= center
            depth[mask] = 1.
            # depth[mask] = 10
            # print(center)
        else:
            print('warning: no depth')
            print(smpl_depth_pth)
            assert 'no depth'
        return torch.FloatTensor(depth[None, ...]),center

    def load_back_normal(self,vid,subject):
        assert vid < 361
        vid = (vid+180) % 360
        normal_pth = os.path.join(self.NORMAL, subject, '%d.png' % vid)
        mask_pth = os.path.join(self.MASK, subject, '%d.png' % vid)

        if os.path.exists(normal_pth):
            normal_img = cv.imread(normal_pth, cv.IMREAD_UNCHANGED)
            normal_img = cv.resize(normal_img, (self.img_w, self.img_h))
            normal_img = np.float32(cv.cvtColor(normal_img, cv.COLOR_BGR2RGB)) / 255
            normal_img = 2.0 * normal_img - 1.0

            mask = Image.open(mask_pth).convert('RGB')
            mask = np.array(mask)[:, :, 0] / 255
            # print(np.unique(mask))
            mask = mask[:,:,None]
            normal_img = normal_img * mask + (1-mask)
            normal_img = torch.FloatTensor(normal_img)
            normal_img = normal_img.permute(2, 0, 1)
        else:
            print('warning: no depth')
            assert 'no depth'

        return normal_img,torch.FloatTensor(mask).permute(2, 0, 1)

class ImgLoader(DataLoader):
    def __init__(self, dataset_dir, img_h, img_w, training=True, train_color=False, testing_res=512,
                 batch_size=4, num_workers=0):
        self.dataset = ImgDataset(dataset_dir, img_h, img_w, training, train_color, testing_res)
        self.batch_size = batch_size

        super(ImgLoader, self).__init__(self.dataset,
                                        batch_size=batch_size,
                                        shuffle=training,
                                        num_workers=num_workers,
                                        worker_init_fn=worker_init_fn,
                                        drop_last=True)


if __name__ == '__main__':
    """tests data loader"""
    dataset_dir = '/data/TwinDom_PerspectRandom_Noisy'
    config.use_joints_sample = True
    loader = ImgLoader(dataset_dir, config.img_h, config.img_w,
                       training=True, train_color=False, batch_size=1, num_workers=2)
    config.use_smpl_depth =True
    for items in loader:
        imgs_tensor = items['rgbd_imgs']
        joints = items['joints']
        pc_joints = items['pc_joints']
        # print(joints.shape,pc_joints.shape)
        break
        # view_bath = items['view_ids']
        # imgs_tensor = imgs_tensor.view(
        #     imgs_tensor.shape[0] * imgs_tensor.shape[1],
        #     imgs_tensor.shape[2],
        #     imgs_tensor.shape[3],
        #     imgs_tensor.shape[4]
        # )
        # print(view_bath)
        # view_bath = view_bath.view(-1)
        # print(view_bath)
        # print(view_bath.shape)
        # print(imgs_tensor.shape)
        # break
    # a = np.random.randn(1, 60, 72)
    # b = a[:, view_bath]
    # print(b.shape)