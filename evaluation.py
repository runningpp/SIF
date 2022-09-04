# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
from pykdtree.kdtree import KDTree
import os
from PIL import Image
import taichi_three as t3
import numpy as np
from taichi_three.transform import *
from tqdm import tqdm
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
class StaticRenderer:
    def __init__(self):
        ti.init(ti.cpu)
        self.scene = t3.Scene()
        self.N = 10

    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            # save_tex.append(model.init_tex)
        ti.init(ti.cpu)
        print('init')
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            # model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            model = t3.StaticModel(self.N, obj=save_obj[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()

    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)

    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        # self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()

    def camera_light(self):

        camera = t3.Camera(res=(512, 512))
        self.scene.add_camera(camera)

        light_dir = np.array([0, 0, 1])
        for l in range(6):
            rotate = np.matmul(rotationX(math.radians(np.random.uniform(-30, 30))),
                               rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            self.scene.add_light(light)

class MeshEvaluator:

    def __init__(self):
        self.render = StaticRenderer()
        # self.render1 = StaticRenderer()

    def set_mesh(self, src_path, tgt_path, scale_factor=1.0, offset=0):
        self.src_mesh = trimesh.load(src_path)
        # print(self.src_mesh.vertices.shape)
        self.tgt_mesh = trimesh.load(tgt_path)
        # print(self.tgt_mesh.vertices.shape)
        self.scale_factor = scale_factor
        self.offset = offset

        self.pred_path = src_path
        self.tgt_path = tgt_path

        obj = t3.readobj(src_path, scale=1)
        # print('lpp', len(self.render.scene.models))
        # self.render.add_model(obj)
        if len(self.render.scene.models) >= 1:
            self.render.modify_model(0, obj)
        else:
            self.render.add_model(obj)

        r_color = np.zeros((obj['vi'].shape[0], 3))
        r_color[:, 0] = 1
        self.render.scene.models[0].modify_color(r_color)

        self.obj = t3.readobj(tgt_path, scale=scale_factor)
        vi = obj['vi']
        median = np.median(vi, axis=0)  # + (np.random.randn(3) - 0.5) * 0.2
        vmin = vi.min(0)
        vmax = vi.max(0)
        median[1] = (vmax[1] * 4 + vmin[1] * 3) / 7

        dis = vmax[1] - vmin[1]
        dis *= 2
        self.ori_vec = np.array([0, 0, dis])
        self.target = median


    def get_chamfer_dist(self, num_samples=10000):
        # Chamfer
        src_surf_pts, idx = trimesh.sample.sample_surface(self.src_mesh, num_samples)
        # self.pointcloud_pred = src_surf_pts.astype(np.float32)
        self.normals_pred = self.src_mesh.face_normals[idx]

        tgt_surf_pts, idx = trimesh.sample.sample_surface(self.tgt_mesh, num_samples)
        # self.pointcloud_gt = tgt_surf_pts.astype(np.float32)
        self.normals_gt = self.tgt_mesh.face_normals[idx]

        _, src_tgt_dist, idx = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)
        self.pred_gt_normals = self.tgt_mesh.face_normals[idx]

        _, tgt_src_dist, idx = trimesh.proximity.closest_point(self.src_mesh, tgt_surf_pts)
        self.gt_pred_normals = self.src_mesh.face_normals[idx]

        src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        tgt_src_dist[np.isnan(tgt_src_dist)] = 0

        self.src_tgt_dist = src_tgt_dist.mean()
        tgt_src_dist = tgt_src_dist.mean()

        chamfer_dist = (self.src_tgt_dist + tgt_src_dist) / 2

        return chamfer_dist

    def get_surface_dist(self, num_samples=10000):
        # P2S
        # src_surf_pts, _ = trimesh.sample.sample_surface(self.src_mesh, num_samples)
        #
        # _, src_tgt_dist, _ = trimesh.proximity.closest_point(self.tgt_mesh, src_surf_pts)
        #
        # src_tgt_dist[np.isnan(src_tgt_dist)] = 0
        #
        # src_tgt_dist = src_tgt_dist.mean()

        return self.src_tgt_dist

    def _get_reproj_normal_error(self, angle):
        p = -15
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, self.ori_vec)
        pos = self.target + fwd

        self.render.scene.models[0].type[None] = 0

        fx, fy = 850, 850
        _cx, _cy = 256, 285

        self.render.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        self.render.scene.cameras[0].set(pos=pos, target=self.target)
        self.render.scene.cameras[0]._init()

        self.render.scene.render()
        camera = self.render.scene.cameras[0]
        self.pred_normal = camera.normal_map.to_numpy().swapaxes(0, 1)[::-1, :]

        if len(self.render.scene.models) >= 1:
            self.render.modify_model(0, self.obj)
        else:
            self.render.add_model(self.obj)

        r_color = np.zeros((self.obj['vi'].shape[0], 3))
        r_color[:, 0] = 1
        self.render.scene.models[0].modify_color(r_color)

        self.render.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        self.render.scene.cameras[0].set(pos=pos, target=self.target)
        self.render.scene.cameras[0]._init()

        self.render.scene.render()
        camera = self.render.scene.cameras[0]
        self.tgt_normal = camera.normal_map.to_numpy().swapaxes(0, 1)[::-1, :]

        error = ((self.pred_normal[:, :, :3] - self.tgt_normal[:, :, :3]) ** 2).mean() * 3
        return error,self.pred_normal,self.tgt_normal

    def get_reproj_normal_error(self, frontal=True, back=True, left=True, right=True, save_demo_img=None):
        # reproj error
        # if save_demo_img is not None, save a visualization at the given path (etc, "./test.png")

        side_cnt = 0
        total_error = 0
        demo_list = []
        if frontal:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(0)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if back:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(180)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if left:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(90)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if right:
            side_cnt += 1
            error, src_normal, tgt_normal = self._get_reproj_normal_error(270)
            total_error += error
            demo_list.append(np.concatenate([src_normal, tgt_normal], axis=0))
        if save_demo_img is not None:
            # print(save_demo_img)
            res_array = np.concatenate(demo_list, axis=1)
            res_img = Image.fromarray((res_array * 255).astype(np.uint8))
            # print(res_array)
            res_img.save(save_demo_img)
        return total_error / side_cnt

    def normal_consistency(self):
        ''' Computes minimal distances of each point in points_src to points_tgt.
        Args:
            points_src (numpy array): source points
            normals_src (numpy array): source normals
            points_tgt (numpy array): target points
            normals_tgt (numpy array): target normals
        '''

        completeness  = (self.normals_gt*self.gt_pred_normals).sum(axis=-1)
        accuracy = (self.normals_pred*self.pred_gt_normals).sum(axis=-1)
        completeness_normals =  completeness.mean()
        accuracy_normals = accuracy.mean()
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )

        return normals_correctness

def eval(results_dir,src_dir,save_dir ='results'):
    # names = sorted(os.listdir(src_dir))

    f_name = os.path.basename(results_dir)
    names_re = sorted(os.listdir(results_dir))
    total_vals = []
    items = []
    os.makedirs(os.path.join(save_dir,f_name),exist_ok=True)
    for j in range(1):
        for i,name in tqdm(enumerate(names_re)):
            if 'smpl' in name:
                # print(name)
                continue
            name_ = name.split('_')[0]
        # for name in names:
            # gt_path = os.path.join(src_dir,name,name[:-2]+'_scaled_meshOnly.ply')
            gt_path = os.path.join(src_dir, name_+'-h', name_+ '_scaled.obj')
            # gt_path = os.path.join(src_dir, name_, name_[:-2] + '_scaled.obj')
            pred_path = os.path.join(results_dir,name)
            evaluator.set_mesh(src_path=pred_path,tgt_path=gt_path,scale_factor=1.0,offset=0.)
            vals = []
            vals.append(1000*evaluator.get_chamfer_dist(num_samples=20000))
            vals.append(1000*evaluator.get_surface_dist(num_samples=20000))
            vals.append(evaluator.normal_consistency())
            # vals.append(4*evaluator.get_reproj_normal_error(
            #     save_demo_img=os.path.join(save_dir,f_name, '%s_normal.png' % (name))))
            item = {
                'name': '%s' % (name),
                'vals': vals
            }
            print(name,vals)
            total_vals.append(vals)
            items.append(item)
    vals = np.array(total_vals).mean(0)
    # np.save(os.path.join(save_dir, f_name, 'item.npy'), np.array(items))
    # np.save(os.path.join(save_dir, f_name, 'vals.npy'), total_vals)
    np.save(os.path.join(save_dir, f_name, 'item.npy'), np.array(items))
    np.save(os.path.join(save_dir, f_name, 'vals.npy'), total_vals)
    # print('chamfer: %.4f  p2s: %.4f nml_consis:%.4f  nml: %.4f' % (vals[0], vals[1], vals[2],vals[3]))
    print('chamfer: %.4f  p2s: %.4f nml_consis:%.4f' % (vals[0], vals[1], vals[2]))
    # print('chamfer: %.4f  p2s: %.4f '% (vals[0], vals[1]))

def eval_smpl(results_dir,src_dir,save_dir ='results'):
    # names = sorted(os.listdir(src_dir))

    names_re = sorted(os.listdir(results_dir))[300:350]
    total_vals = []
    items = []

    for j in range(1):
        for i,name in tqdm(enumerate(names_re)):

        # for name in names:
            # gt_path = os.path.join(src_dir,name,name[:-2]+'_scaled_meshOnly.ply')
            gt_path = os.path.join(src_dir, name, name[:-2]+ '_scaled.obj')
            # gt_path = os.path.join(src_dir, name_, name_[:-2] + '_scaled.obj')
            pred_path = os.path.join(results_dir,name,'smpl_mesh_scaled.obj')
            evaluator.set_mesh(src_path=pred_path,tgt_path=gt_path,scale_factor=1.0,offset=0.)
            vals = []
            vals.append(1000*evaluator.get_chamfer_dist(num_samples=6000))
            vals.append(1000*evaluator.get_surface_dist(num_samples=6000))
            vals.append(evaluator.normal_consistency())
            # vals.append(4*evaluator.get_reproj_normal_error(
            #     save_demo_img=os.path.join(save_dir,f_name, '%s_normal.png' % (name))))
            item = {
                'name': '%s' % (name),
                'vals': vals
            }
            print(name,vals)
            total_vals.append(vals)
            items.append(item)
    vals = np.array(total_vals).mean(0)
    # np.save(os.path.join(save_dir, f_name, 'item.npy'), np.array(items))
    # np.save(os.path.join(save_dir, f_name, 'vals.npy'), total_vals)
    # np.save(os.path.join(save_dir, f_name, 'item.npy'), np.array(items))
    # np.save(os.path.join(save_dir, f_name, 'vals.npy'), total_vals)
    # print('chamfer: %.4f  p2s: %.4f nml_consis:%.4f  nml: %.4f' % (vals[0], vals[1], vals[2],vals[3]))
    print('chamfer: %.4f  p2s: %.4f nml_consis:%.4f' % (vals[0], vals[1], vals[2]))
    # print('chamfer: %.4f  p2s: %.4f '% (vals[0], vals[1]))

if __name__ == '__main__':
    import torch
    torch.manual_seed(414717)
    np.random.seed(414717)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target_path', type=str, default='/home/lpp/data/TwinDom_PerspectRandom_Noisy')
    parser.add_argument('-r', '--result_path', type=str, default='results/rgb_icon_filter7_add_z_epoch25')
    parser.add_argument('-s', '--save_path', type=str, default='/home/lpp/results/result')
    args = parser.parse_args()

    evaluator = MeshEvaluator()

    root = args.target_path
    pred_root = args.result_path
    save_dir = args.save_path
    eval(results_dir=pred_root,src_dir=root,save_dir=save_dir)
    # eval_smpl(results_dir=pred_root, src_dir=root, save_dir=save_dir)

'''
pifu_ori_100_epoch_12
chamfer: 3.7223  p2s: 3.5725 nml_consis:0.8760  nml: 0.0075 
pifu_w_ptsnml_normal_100_
chamfer: 6.0667  p2s: 5.3375 nml_consis:0.8754  nml: 0.0079




###rgbd_joints_cutoff_joints_sample_epoch30:

rgbd_joints_cut0ff_epoch15:

rgbd_smpl_double_epoch25
chamfer: 7.1312  p2s: 6.6023 nml_consis:0.8215

rgbd_spconv_smpl_epoch25_
chamfer: 8.0148  p2s: 7.8002 nml_consis:0.7868

rgbd_w_smpl_depth_b_normal_epoch25
chamfer: 6.5122  p2s: 7.5178 nml_consis:0.7856

/rgbd_w_smpl_joints_verts_b_normal_epoch25
chamfer: 8.1108  p2s: 8.5990 nml_consis:0.758

# rgbd_w_smpl_depth_b_normal_v1_epoch25
# chamfer: 4.2427  p2s: 4.2706 nml_consis:0.8456

rgbd_w_smpl_depth_normal_joints_v1_epoch25
chamfer: 3.0672  p2s: 3.0166 nml_consis:0.8712

rgbd_w_smpl_depth_normal_joints_w_vis_epoch25
chamfer: 3.3042  p2s: 3.5229 nml_consis:0.8274

Twindom_smpl_915
chamfer: 10.3470  p2s: 9.6107 nml_consis:0.7954

rgb_pami_hrnet256_ve32_epoch25
5.5982  p2s: 5.3239 nml_consis:0.8573

rgb_pami_hrnet64_ve32_sematic_vol_epoch25
chamfer: 5.5413  p2s: 5.2403 nml_consis:0.8570

rgb_icon_filter7_epoch25
chamfer: 4.9698  p2s: 5.3008 nml_consis:0.7967

rgb_icon_filter7_add_z_epoch25
chamfer: 4.5096  p2s: 4.6425 nml_consis:0.8194
'''