from __future__ import division, print_function

from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from skimage import measure
import datetime
import scipy.io as sio
import time
import multiprocessing

# from network.fusion_model_v1 import PIFUHrnet
from network.arch_icon import PIFUHrnet
import config
from util import lr_schedule
import util.utils as util
import util.obj_io as obj_io
from dataloader.dataloader_icon import ImgLoader
# import spconv
# from apex import amp

log = util.logger.write
now = datetime.datetime.now()

class Trainer(object):
    def __init__(self, use_hrnet=True,use_filter=False,use_mlp_icon=True):
        super(Trainer, self).__init__()
        if use_hrnet:
            # self.pifu = PIFUHrnet().to(config.device)
            self.pifu = PIFUHrnet(use_filter,use_mlp_icon).to(config.device)

        # self.mse = nn.MSELoss().to(config.device)
        self.bce = nn.BCELoss().to(config.device)
        # self.bce = nn.BCEWithLogitsLoss().to(config.device)
        # self.optm = torch.optim.Adam(
        #     params=self.pifu.parameters(),
        #     lr=float(config.learning_rate)
        # )
        self.optm = torch.optim.Adam(
            params=self.pifu.parameters(),
            lr=float(config.learning_rate)
        )

        self.schedule = lr_schedule.get_learning_rate_schedules(
            'Step', Initial=config.learning_rate, Interval=10, Factor=0.1)

        # self.pifu,self.optm = amp.initialize(self.pifu,self.optm,opt_level='O1')
        log('#trainable_params = %d' %
            sum(p.numel() for p in self.pifu.parameters() if p.requires_grad))

    def reshape_multiview_imgs(self, imgs_tensor):
        imgs_tensor = imgs_tensor.view(
            imgs_tensor.shape[0] * imgs_tensor.shape[1],
            imgs_tensor.shape[2],
            imgs_tensor.shape[3],
            imgs_tensor.shape[4]
        )
        return imgs_tensor

    def reshape_multiview_tensors(self, rgbd_imgs_tensor,  pts_tensor):
        rgbd_imgs_tensor = self.reshape_multiview_imgs(rgbd_imgs_tensor)
        # depth_masks_tensor = self.reshape_multiview_imgs(depth_masks_tensor)

        pts_tensor = pts_tensor.view(
            pts_tensor.shape[0] * pts_tensor.shape[1],
            pts_tensor.shape[2],
            pts_tensor.shape[3])
        return rgbd_imgs_tensor, pts_tensor

    def train_network(self, dataset_dir, output_dir, epoch_num,
                      pre_trained_model=None, previous_optimizer=None, start_epoch=0):
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime('%b%d_%H_%M_%S')
        log_dir = os.path.join(output_dir, 'runs', current_time)
        writer = SummaryWriter(log_dir=log_dir)
        # writer = SummaryWriter()

        # worker_num = min(multiprocessing.cpu_count(),
        #                  max(config.batch_size * config.view_num, 6))
        # # debug
        worker_num = config.work_num

        print("num_workers of DataLoader: {}".format(worker_num))
        loader = ImgLoader(dataset_dir, config.img_h, config.img_w,
                           training=True, batch_size=config.batch_size, num_workers=worker_num)


        # backup current code
        # current_folder = os.path.split(os.path.abspath(__file__))[0]
        # tar_file = os.path.join(output_dir, 'code_bk_%s.tar.gz' %
        #                         now.strftime('%Y_%m_%d_%H_%M_%S'))
        # Trainer.create_code_snapshot(current_folder, tar_file)

        # continue from previous run
        self.pifu.train()
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
        if previous_optimizer is not None:
            self.load_optimizer(previous_optimizer)

        # batch_num = len(loader.dataset) // config.render_view_number
        # print('batch_num: {}'.format(batch_num))
        batch_num = len(loader.dataset) // config.batch_size
        print('batch_num: {}'.format(batch_num))
        # batch_num = len(loader)
        # print(batch_num)
        for epoch in range(start_epoch, epoch_num):
            # log('Training epoch: %d' % epoch)
            epoch_loss = 0.0
            self.update_learning_rate(epoch)
            tstart = time.time()
            for i, items in enumerate(loader):
                # if i < 2700:
                #     continue
                # t0 = time.time()
                # print('data loader is', t0 - tstart)

                normal_batch = items['depth_normal'].to(config.device,non_blocking=True)[:,:,1:] #B,2,1+3,H,W
                # print(normal_batch.shape)
                # mask_batch = items['mask'].to(config.device,non_blocking=True)
                pts_batch = items['pts'].to(config.device, non_blocking=True)
                gt_batch = items['pts_label'].to(config.device, non_blocking=True)
                # print(gt_batch.reshape(-1))

                pc_joints_smpl = items['pc_smpl_semantic'].to(config.device, non_blocking=True) #B,N,7
                pts_vis_batch = items['pts_vis'].to(config.device, non_blocking=True) #B,N,1
                # print(pts_vis_batch.shape,pc_joints_smpl.shape,)

                normal_batch, pts_batch = (
                    self.reshape_multiview_tensors(normal_batch, pts_batch))
                # print(rgbd_imgs_batch.shape)
                # print(pts_batch.shape)
                # print(gt_batch.shape)
                # t1 = time.time()
                # print('time1', t1 - t0)
                self.zero_grad()
                _, loss, logs = self.forward(
                    normal_batch, pts_batch, gt_batch,
                    training=True,pc_smpl_semantic=pc_joints_smpl,pts_vis=pts_vis_batch)

                # t2 = time.time()
                # print('forward time is', t2 - t1)

                writer.add_scalar('Loss/Batch', loss, epoch * batch_num + i)
                epoch_loss += loss
                self.optm_one_step()
                # print('backward time is', time.time() - t2)
                log('Epoch %d, Batch %d: ' % (epoch, i) +
                    ''.join([('%s=%f ' % (k, logs[k])) for k in sorted(logs.keys())]))
                # with open(log_file, 'a') as fp:
                #     s = '%d %d ' % (epoch, i)
                #     s += ''.join([('%f ' % logs[k])
                #                   for k in sorted(logs.keys())])
                #     fp.write(s + '\n')
                # if i == batch_num:
                #     break
                if i > 0 and i % 100 == 0:
                    model_folder = os.path.join(output_dir, 'epoch_%03d' % epoch)
                    util.safe_mkdir(model_folder)
                    self.save(model_folder)
                    optm_path = os.path.join(output_dir, 'optm.pth')
                    torch.save({'optimizer': self.optm.state_dict()}, optm_path)

            # tend = time.time()
            # print('batch time: {}'.format(tend - tstart))

            epoch_loss = epoch_loss / batch_num
            writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
            # if (((batch_num == 1) and (epoch % 50 == 0)) or (batch_num > 1 and epoch % 1 == 0)):
            model_folder = os.path.join(output_dir, 'epoch_%03d' % epoch)
            util.safe_mkdir(model_folder)
            self.save(model_folder)
            optm_path = os.path.join(output_dir, 'optm.pth')
            torch.save({'optimizer': self.optm.state_dict()}, optm_path)
            log('End of epoch {}.'.format(epoch))
            self.save(output_dir)
        writer.close()
        log('End of training. ')

    def eval_network(self, dataset_dir, model_dir, output_dir):
        # from dataloader.dataloader_v1 import ImgLoader
        loader = ImgLoader(dataset_dir, config.img_h, config.img_w,
                           training=False, testing_res=config.testing_res, batch_size=1)
        util.safe_mkdir(output_dir)
        self.pifu.eval()
        self.load_model(model_dir)

        print(len(loader.dataset.data_list))
        for i in range(len(loader.dataset.data_list)):
            # [i,config.render_view_number)]
            # view = i % config.render_view_number
            view = i % 10 * 3
            # subject_id = i / config.render_view_number
            # if subject_id <= 4:
            #     continue

            if (view % 1 == 0):

                folder, _ = os.path.split(loader.dataset.data_list[i])
                _, subject_name = os.path.split(folder)
                # if os.path.exists(os.path.join(
                #         output_dir,
                #         '{}_vnum{}_view{}_res{}.obj'.format(subject_name[:-2], config.view_num, view,
                #                                             config.testing_res))):
                #     print('has existed',
                #           '{}_vnum{}_view{}_res{}.obj'.format(subject_name[:-2], config.view_num, view,
                #                                               config.testing_res))
                #     continue
                items = loader.dataset.__getitem__(i)

                pts_ov = items['pts_flag']

                normal_batch = items['depth_normal'].unsqueeze(0)[:,:,1:]
                # mask_batch = items['mask'].unsqueeze(0)
                pts_batch = items['pts'].unsqueeze(0)

                # print(gt_batch.reshape(-1))

                # pts_normal_batch = items['pts_normal'].unsqueeze(0)
                pc_joints_smpl = items['pc_smpl_semantic'].unsqueeze(0)
                pts_vis_batch = items['pts_vis'].unsqueeze(0)

                # print(pc_joints_batch.dtype)
                # pc_joints_batch = torch.norm(pc_joints_batch, dim=-1, keepdim=True)

                # rgbd_imgs_batch, depth_masks_batch, pts_batch = (
                #     self.reshape_multiview_tensors(rgbd_imgs_batch, depth_masks_batch, pts_batch))
                normal_batch, pts_batch = (
                    self.reshape_multiview_tensors(normal_batch, pts_batch))

                t1 = time.time()

                outputs, _, _=self.forward(
                    normal_batch, pts_batch, None,
                    training=False, pc_smpl_semantic=pc_joints_smpl,  pts_vis=pts_vis_batch)
                print('forward time: {}'.format(time.time() - t1))

                t2 = time.time()
                # print(pts_ov.shape,outputs[-1].shape)
                pts_ov[pts_ov > 0] = outputs[-1].view(-1)
                # pts_ov = outputs[-1].view(-1)
                pts_ov = torch.reshape(pts_ov, (config.testing_res, config.testing_res, config.testing_res))
                print('generate pts_ov time: {}'.format(time.time() - t2))

                t4 = time.time()
                voxel_size = 1 / config.testing_res
                pts_ov = pts_ov.cpu().numpy()
                # print('pts_ov.min(): {}, pts_ov.max(): {}'.format(pts_ov.min(), pts_ov.max()))
                vertices, faces, normals, _ = measure.marching_cubes_lewiner(pts_ov, 0.5,
                                                                             (voxel_size, voxel_size, voxel_size))
                print('maching cubes time: {}'.format(time.time() - t4))

                t5 = time.time()

                mesh = dict()
                mesh['v'] = vertices - \
                            np.ones((1, 3)) * 0.5 + np.ones((1, 3)) * \
                            (1 / config.testing_res / 2)
                # print(vertices.shape)
                mesh['f'] = faces
                mesh['f'] = mesh['f'][:, (1, 0, 2)]
                mesh['vn'] = normals
                obj_io.save_obj_data(mesh, os.path.join(
                    output_dir,
                    '{}_vnum{}_view{}_res{}.obj'.format(subject_name[:-2], config.view_num, loader.dataset.view,
                                                        config.testing_res)))
                print('save mesh obj time: {}'.format(time.time() - t5))

                # release memory
                del items, pts_batch, outputs, vertices, pts_ov, normal_batch,pc_joints_smpl,pts_vis_batch
                # except:
                #     print(loader.dataset.view,'out of memory')

        # # debug: marching cube GT volume
        # gt_volume = sio.loadmat('F:/Code/RGBDPIFu/Dataset/training_dataset_overfitting_test_full_sampling2/126111533418533-h/volumes.mat')['volume']
        # gt_voxel_size = 1 / 800
        # vertices, faces, normals, _ = measure.marching_cubes_lewiner(gt_volume, 0.5, (gt_voxel_size, gt_voxel_size, gt_voxel_size))
        # mesh = dict()
        # mesh['v'] = vertices - np.ones((1,3))*0.5
        # # print(vertices.shape)
        # mesh['f'] = faces
        # mesh['f'] = mesh['f'][:, (1, 0, 2)]
        # mesh['vn'] = normals
        # obj_io.save_obj_data(mesh, os.path.join(output_dir, 'gt_mesh.obj'))
        # os._exit(0)

    def forward(self, rgbd_imgs_batch,  pts, gt, training=True, pc_smpl_semantic=None,pts_vis=None):
        # print(rgbd_imgs_batch.dtype,masks.dtype,masks.dtype,pts.dtype,pc_joints.dtype,gt.dtype)
        # normals, pts, pc_normal, pc_smpl_senmatics = None, pts_vis_flag = None
        outputs = self.pifu(rgbd_imgs_batch, pts,pc_smpl_semantic, pts_vis)
        if training:
            recon_loss = 0
            for output in outputs:
                # recon_loss += self.mse(output, gt)
                recon_loss += self.bce(output, gt)
            logs = {
                'l_recon': recon_loss.item(),
            }
            # with amp.scale_loss(recon_loss,self.optm) as scaled_loss:
            #     scaled_loss.backward()
            recon_loss.backward()
            return outputs, recon_loss, logs
        else:
            return outputs, None, None

    def zero_grad(self):
        self.pifu.zero_grad()

    def optm_one_step(self):
        self.optm.step()

    def update_learning_rate(self, epoch):
        lr = self.schedule.get_learning_rate(epoch)
        print('learning rate ', lr)
        for param_group in self.optm.param_groups:
            param_group['lr'] = lr

    def load_model(self, model_folder):
        def load_data_from_pth(path_):
            if torch.cuda.is_available():
                data_ = torch.load(path_)
            else:
                data_ = torch.load(path_,
                                   map_location=lambda storage, loc: storage)
            return data_

        pifu_path = os.path.join(model_folder, 'pifu.pth')
        if os.path.exists(pifu_path):
            log('Loading generator from ' + pifu_path)
            data = load_data_from_pth(pifu_path)
            # self.pifu.load_state_dict(data['generator'])
            save_model = data['generator']
            model_dict = self.pifu.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            print(state_dict.keys())
            model_dict.update(state_dict)
            self.pifu.load_state_dict(model_dict)
        else:
            raise FileNotFoundError("Cannot find pifu.pth in folder: {}".format(model_folder))

    def load_optimizer(self, model_folder):
        def load_data_from_pth(path_):
            if torch.cuda.is_available():
                data_ = torch.load(path_)
            else:
                data_ = torch.load(path_,
                                   map_location=lambda storage, loc: storage)
            return data_

        optm_path = os.path.join(model_folder, 'optm.pth')
        if os.path.exists(optm_path):
            log('Loading generator from ' + optm_path)
            data = load_data_from_pth(optm_path)
            self.optm.load_state_dict(data['optimizer'])
        else:
            raise FileNotFoundError("Cannot find optm.pth in folder: {}".format(model_folder))

    def save(self, model_folder):
        log('Saving to ' + model_folder)
        pifu_path = os.path.join(model_folder, 'pifu.pth')
        torch.save({'generator': self.pifu.state_dict()}, pifu_path)
        # optm_path = os.path.join(model_folder, 'optm.pth')
        # torch.save({'optimizer': self.optm.state_dict()}, optm_path)

    @staticmethod
    def create_code_snapshot(root, dst_path,
                             extensions=(".py", ".h", ".cpp", ".cu", ".cc", ".cuh", ".json", ".sh", ".bat"),
                             exclude=()):
        """Creates tarball with the source code"""
        import tarfile
        from pathlib import Path

        with tarfile.open(str(dst_path), "w:gz") as tar:
            for path in Path(root).rglob("*"):
                if '.git' in path.parts:
                    continue
                exclude_flag = False
                if len(exclude) > 0:
                    for k in exclude:
                        if k in path.parts:
                            exclude_flag = True
                if exclude_flag:
                    continue
                if path.suffix.lower() in extensions:
                    tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


class Trainer1(object):
    def __init__(self, use_hrnet=True):
        super(Trainer, self).__init__()
        if use_hrnet:
            # self.pifu = PIFUHrnet().to(config.device)
            self.pifu = PIFUHrnet1().to(config.device)

        # self.mse = nn.MSELoss().to(config.device)
        self.bce = nn.BCELoss().to(config.device)
        # self.bce = nn.BCEWithLogitsLoss().to(config.device)
        # self.optm = torch.optim.Adam(
        #     params=self.pifu.parameters(),
        #     lr=float(config.learning_rate)
        # )
        self.optm = torch.optim.Adam(
            params=self.pifu.parameters(),
            lr=float(config.learning_rate)
        )

        self.schedule = lr_schedule.get_learning_rate_schedules(
            'Step', Initial=config.learning_rate, Interval=10, Factor=0.1)

        # self.pifu,self.optm = amp.initialize(self.pifu,self.optm,opt_level='O1')
        log('#trainable_params = %d' %
            sum(p.numel() for p in self.pifu.parameters() if p.requires_grad))

    def reshape_multiview_imgs(self, imgs_tensor):
        imgs_tensor = imgs_tensor.view(
            imgs_tensor.shape[0] * imgs_tensor.shape[1],
            imgs_tensor.shape[2],
            imgs_tensor.shape[3],
            imgs_tensor.shape[4]
        )
        return imgs_tensor

    def reshape_multiview_tensors(self, rgbd_imgs_tensor, depth_masks_tensor, pts_tensor):
        rgbd_imgs_tensor = self.reshape_multiview_imgs(rgbd_imgs_tensor)
        depth_masks_tensor = self.reshape_multiview_imgs(depth_masks_tensor)

        pts_tensor = pts_tensor.view(
            pts_tensor.shape[0] * pts_tensor.shape[1],
            pts_tensor.shape[2],
            pts_tensor.shape[3])
        return rgbd_imgs_tensor, depth_masks_tensor, pts_tensor

    def train_network(self, dataset_dir, output_dir, epoch_num, learning_rate,
                      pre_trained_model=None, previous_optimizer=None, start_epoch=0):
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.datetime.now().strftime('%b%d_%H_%M_%S')
        log_dir = os.path.join(output_dir, 'runs', current_time)
        writer = SummaryWriter(log_dir=log_dir)
        # writer = SummaryWriter()

        # worker_num = min(multiprocessing.cpu_count(),
        #                  max(config.batch_size * config.view_num, 6))
        # # debug
        worker_num = config.work_num

        print("num_workers of DataLoader: {}".format(worker_num))
        loader = ImgLoader(dataset_dir, config.img_h, config.img_w,
                           training=True, batch_size=config.batch_size, num_workers=worker_num)


        # backup current code
        # current_folder = os.path.split(os.path.abspath(__file__))[0]
        # tar_file = os.path.join(output_dir, 'code_bk_%s.tar.gz' %
        #                         now.strftime('%Y_%m_%d_%H_%M_%S'))
        # Trainer.create_code_snapshot(current_folder, tar_file)

        # continue from previous run
        self.pifu.train()
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
        if previous_optimizer is not None:
            self.load_optimizer(previous_optimizer)

        # batch_num = len(loader.dataset) // config.render_view_number
        # print('batch_num: {}'.format(batch_num))
        batch_num = len(loader.dataset) // config.batch_size
        print('batch_num: {}'.format(batch_num))
        # batch_num = len(loader)
        # print(batch_num)
        for epoch in range(start_epoch, epoch_num):
            # log('Training epoch: %d' % epoch)
            epoch_loss = 0.0
            self.update_learning_rate(epoch)
            tstart = time.time()
            for i, items in enumerate(loader):
                # if i < 2700:
                #     continue
                # t0 = time.time()
                # print('data loader is', t0 - tstart)
                rgbd_imgs_batch = items['rgbd_imgs'].to(config.device, non_blocking=True)
                depth_masks_batch = items['depth_masks_dt'].to(config.device, non_blocking=True)
                # depth_normal_batch = items['depth_normal'].to(config.device,non_blocking=True)
                # mask_batch = items['mask'].to(config.device,non_blocking=True)
                pts_batch = items['pts'].to(config.device, non_blocking=True)
                gt_batch = items['pts_label'].to(config.device, non_blocking=True)

                back_depth_normal = items['back_depth_normal'].to(config.device,non_blocking=True)  #B,4,h,w
                back_pts = items['pts_back'].to(config.device,non_blocking=True)  #B,N,3
                if config.use_joints_sample:
                    pc_joints_batch = items['pc_joints'].to(config.device, non_blocking=True)  #batch,pts_num,jts_num,3
                    # print(pc_joints_batch.dtype)
                    pc_joints_batch = torch.norm(pc_joints_batch,dim=-1,keepdim=True)
                else:
                    pc_joints_batch =None

                if config.use_smpl_senmantic:
                    pc_joints_smpl = items['pc_smpl_semantic'].to(config.device, non_blocking=True) #B,N,4
                    if config.only_use_smpl_sign:
                        pc_joints_smpl = pc_joints_smpl[:, :, :1]
                else:
                    pc_joints_smpl=None
                # if config.use_smpl_depth:
                #     pc_smpl_psdf = items['pc_smpl_depth_sdf'].to(config.device,non_blocking=True) #B,N


                rgbd_imgs_batch, depth_masks_batch, pts_batch = (
                    self.reshape_multiview_tensors(rgbd_imgs_batch, depth_masks_batch, pts_batch))
                # rgbd_imgs_batch, depth_masks_batch, pts_batch = (
                #     self.reshape_multiview_tensors(depth_normal_batch, mask_batch, pts_batch))
                # print(rgbd_imgs_batch.shape)
                # print(pts_batch.shape)
                # print(gt_batch.shape)
                # t1 = time.time()
                # print('time1', t1 - t0)
                self.zero_grad()
                _, loss, logs = self.forward(
                    rgbd_imgs_batch, depth_masks_batch, pts_batch, back_depth_normal, back_pts, gt_batch,
                    training=True,pc_smpl_semantic=pc_joints_smpl,pc_joints=pc_joints_batch)

                # t2 = time.time()
                # print('forward time is', t2 - t1)

                writer.add_scalar('Loss/Batch', loss, epoch * batch_num + i)
                epoch_loss += loss
                self.optm_one_step()
                # print('backward time is', time.time() - t2)
                log('Epoch %d, Batch %d: ' % (epoch, i) +
                    ''.join([('%s=%f ' % (k, logs[k])) for k in sorted(logs.keys())]))
                # with open(log_file, 'a') as fp:
                #     s = '%d %d ' % (epoch, i)
                #     s += ''.join([('%f ' % logs[k])
                #                   for k in sorted(logs.keys())])
                #     fp.write(s + '\n')
                # if i == batch_num:
                #     break
                if i > 0 and i % 100 == 0:
                    model_folder = os.path.join(output_dir, 'epoch_%03d' % epoch)
                    util.safe_mkdir(model_folder)
                    self.save(model_folder)
                    optm_path = os.path.join(output_dir, 'optm.pth')
                    torch.save({'optimizer': self.optm.state_dict()}, optm_path)

            # tend = time.time()
            # print('batch time: {}'.format(tend - tstart))

            epoch_loss = epoch_loss / batch_num
            writer.add_scalar('Loss/Epoch', epoch_loss, epoch)
            # if (((batch_num == 1) and (epoch % 50 == 0)) or (batch_num > 1 and epoch % 1 == 0)):
            model_folder = os.path.join(output_dir, 'epoch_%03d' % epoch)
            util.safe_mkdir(model_folder)
            self.save(model_folder)
            optm_path = os.path.join(output_dir, 'optm.pth')
            torch.save({'optimizer': self.optm.state_dict()}, optm_path)
            log('End of epoch {}.'.format(epoch))
            self.save(output_dir)
        writer.close()
        log('End of training. ')

    def eval_network(self, dataset_dir, model_dir, output_dir):
        # from dataloader.dataloader_v1 import ImgLoader
        loader = ImgLoader(dataset_dir, config.img_h, config.img_w,
                           training=False, testing_res=config.testing_res, batch_size=1)
        util.safe_mkdir(output_dir)
        self.pifu.eval()
        self.load_model(model_dir)

        print(len(loader.dataset.data_list))
        for i in range(len(loader.dataset.data_list)):
            # [i,config.render_view_number)]
            # view = i % config.render_view_number
            view = i % 10 * 3
            # subject_id = i / config.render_view_number
            # if subject_id <= 4:
            #     continue

            if (view % 1 == 0):

                folder, _ = os.path.split(loader.dataset.data_list[i])
                _, subject_name = os.path.split(folder)
                # if os.path.exists(os.path.join(
                #         output_dir,
                #         '{}_vnum{}_view{}_res{}.obj'.format(subject_name[:-2], config.view_num, view,
                #                                             config.testing_res))):
                #     print('has existed',
                #           '{}_vnum{}_view{}_res{}.obj'.format(subject_name[:-2], config.view_num, view,
                #                                               config.testing_res))
                #     continue
                items = loader.dataset.__getitem__(i)
                rgbd_imgs_batch = (items['rgbd_imgs'].unsqueeze(0))
                depth_masks_batch = (items['depth_masks_dt'].unsqueeze(0))
                pts_batch = items['pts'].unsqueeze(0)
                pts_ov = items['pts_flag']
                back_depth_normal = items['back_depth_normal'].unsqueeze(0)  # B,4,h,w
                back_pts = items['pts_back'].unsqueeze(0)
                if config.use_smpl_senmantic:
                    pc_joints_smpl = items['pc_smpl_semantic'].unsqueeze(0).to(config.device) #B,N,4
                    if config.only_use_smpl_sign:
                        pc_joints_smpl =  pc_joints_smpl[:,:,:1]
                else:
                    pc_joints_smpl = None
                if config.use_joints_sample:
                    pc_joints_batch = items['pc_joints'].unsqueeze(0)  # batch,pts_num,jts_num,3
                    # print(pc_joints_batch.dtype)
                    pc_joints_batch = torch.norm(pc_joints_batch, dim=-1, keepdim=True)
                else:
                    pc_joints_batch = None
                # print(pc_joints_batch.dtype)
                # pc_joints_batch = torch.norm(pc_joints_batch, dim=-1, keepdim=True)

                rgbd_imgs_batch, depth_masks_batch, pts_batch = (
                    self.reshape_multiview_tensors(rgbd_imgs_batch, depth_masks_batch, pts_batch))

                t1 = time.time()
                outputs, _, _ = self.forward(
                    rgbd_imgs_batch, depth_masks_batch, pts_batch, back_depth_normal, back_pts, None,
                    training=False,pc_smpl_semantic=pc_joints_smpl,pc_joints=pc_joints_batch)
                print('forward time: {}'.format(time.time() - t1))

                t2 = time.time()
                # print(pts_ov.shape,outputs[-1].shape)
                pts_ov[pts_ov > 0] = outputs[-1].view(-1)
                # pts_ov = outputs[-1].view(-1)
                pts_ov = torch.reshape(pts_ov, (config.testing_res, config.testing_res, config.testing_res))
                print('generate pts_ov time: {}'.format(time.time() - t2))

                t4 = time.time()
                voxel_size = 1 / config.testing_res
                pts_ov = pts_ov.cpu().numpy()
                # print('pts_ov.min(): {}, pts_ov.max(): {}'.format(pts_ov.min(), pts_ov.max()))
                vertices, faces, normals, _ = measure.marching_cubes_lewiner(pts_ov, 0.5,
                                                                             (voxel_size, voxel_size, voxel_size))
                print('maching cubes time: {}'.format(time.time() - t4))

                t5 = time.time()

                mesh = dict()
                mesh['v'] = vertices - \
                            np.ones((1, 3)) * 0.5 + np.ones((1, 3)) * \
                            (1 / config.testing_res / 2)
                # print(vertices.shape)
                mesh['f'] = faces
                mesh['f'] = mesh['f'][:, (1, 0, 2)]
                mesh['vn'] = normals
                obj_io.save_obj_data(mesh, os.path.join(
                    output_dir,
                    '{}_vnum{}_view{}_res{}.obj'.format(subject_name[:-2], config.view_num, loader.dataset.view,
                                                        config.testing_res)))
                print('save mesh obj time: {}'.format(time.time() - t5))

                # release memory
                del items, pts_batch, outputs, vertices, pts_ov, depth_masks_batch, rgbd_imgs_batch
                # except:
                #     print(loader.dataset.view,'out of memory')

        # # debug: marching cube GT volume
        # gt_volume = sio.loadmat('F:/Code/RGBDPIFu/Dataset/training_dataset_overfitting_test_full_sampling2/126111533418533-h/volumes.mat')['volume']
        # gt_voxel_size = 1 / 800
        # vertices, faces, normals, _ = measure.marching_cubes_lewiner(gt_volume, 0.5, (gt_voxel_size, gt_voxel_size, gt_voxel_size))
        # mesh = dict()
        # mesh['v'] = vertices - np.ones((1,3))*0.5
        # # print(vertices.shape)
        # mesh['f'] = faces
        # mesh['f'] = mesh['f'][:, (1, 0, 2)]
        # mesh['vn'] = normals
        # obj_io.save_obj_data(mesh, os.path.join(output_dir, 'gt_mesh.obj'))
        # os._exit(0)

    def forward(self, rgbd_imgs_batch, masks, pts, back_imgs,pts_back,gt,training=True,pc_smpl_semantic=None,pc_joints=None):
        # print(rgbd_imgs_batch.dtype,masks.dtype,masks.dtype,pts.dtype,pc_joints.dtype,gt.dtype)
        outputs = self.pifu(rgbd_imgs_batch, masks, pts,back_imgs,pts_back,pc_smpl_semantic,pc_joints)
        if training:
            recon_loss = 0
            for output in outputs:
                # recon_loss += self.mse(output, gt)
                recon_loss += self.bce(output, gt)
            logs = {
                'l_recon': recon_loss.item(),
            }
            # with amp.scale_loss(recon_loss,self.optm) as scaled_loss:
            #     scaled_loss.backward()
            recon_loss.backward()
            return outputs, recon_loss, logs
        else:
            return outputs, None, None

    def zero_grad(self):
        self.pifu.zero_grad()

    def optm_one_step(self):
        self.optm.step()

    def update_learning_rate(self, epoch):
        lr = self.schedule.get_learning_rate(epoch)
        print('learning rate ', lr)
        for param_group in self.optm.param_groups:
            param_group['lr'] = lr

    def load_model(self, model_folder):
        def load_data_from_pth(path_):
            if torch.cuda.is_available():
                data_ = torch.load(path_)
            else:
                data_ = torch.load(path_,
                                   map_location=lambda storage, loc: storage)
            return data_

        pifu_path = os.path.join(model_folder, 'pifu.pth')
        if os.path.exists(pifu_path):
            log('Loading generator from ' + pifu_path)
            data = load_data_from_pth(pifu_path)
            # self.pifu.load_state_dict(data['generator'])
            save_model = data['generator']
            model_dict = self.pifu.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            print(state_dict.keys())
            model_dict.update(state_dict)
            self.pifu.load_state_dict(model_dict)
        else:
            raise FileNotFoundError("Cannot find pifu.pth in folder: {}".format(model_folder))

    def load_optimizer(self, model_folder):
        def load_data_from_pth(path_):
            if torch.cuda.is_available():
                data_ = torch.load(path_)
            else:
                data_ = torch.load(path_,
                                   map_location=lambda storage, loc: storage)
            return data_

        optm_path = os.path.join(model_folder, 'optm.pth')
        if os.path.exists(optm_path):
            log('Loading generator from ' + optm_path)
            data = load_data_from_pth(optm_path)
            self.optm.load_state_dict(data['optimizer'])
        else:
            raise FileNotFoundError("Cannot find optm.pth in folder: {}".format(model_folder))

    def save(self, model_folder):
        log('Saving to ' + model_folder)
        pifu_path = os.path.join(model_folder, 'pifu.pth')
        torch.save({'generator': self.pifu.state_dict()}, pifu_path)
        # optm_path = os.path.join(model_folder, 'optm.pth')
        # torch.save({'optimizer': self.optm.state_dict()}, optm_path)

    @staticmethod
    def create_code_snapshot(root, dst_path,
                             extensions=(".py", ".h", ".cpp", ".cu", ".cc", ".cuh", ".json", ".sh", ".bat"),
                             exclude=()):
        """Creates tarball with the source code"""
        import tarfile
        from pathlib import Path

        with tarfile.open(str(dst_path), "w:gz") as tar:
            for path in Path(root).rglob("*"):
                if '.git' in path.parts:
                    continue
                exclude_flag = False
                if len(exclude) > 0:
                    for k in exclude:
                        if k in path.parts:
                            exclude_flag = True
                if exclude_flag:
                    continue
                if path.suffix.lower() in extensions:
                    tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


