from __future__ import print_function, absolute_import, division
import config
import os
# import pynvml
import torch
import numpy as np
import time

from trainer.trainer_sif import Trainer


# pynvml.nvmlInit()
# deviceCount = pynvml.nvmlDeviceGetCount()
# use_gpu_id = deviceCount - 1
# print('Found %d GPU(s). GPU No.%d will be used. ' % (deviceCount, use_gpu_id))
os.environ["CUDA_VISIBLE_DEVICES"] = ("%d" % 0)

def main_train_geometry():
    torch.manual_seed(31359)
    np.random.seed(31359)
    config.input_mode = 'normal'
    config.num_out = 32
    config.use_joints_sample = True
    config.use_cutoff = True
    config.epoch_num = 26
    config.batch_size = 12
    config.work_num = 12
    config.use_smpl_senmantic = False
    config.only_use_smpl =  False
    config.only_use_smpl_sign = False
    config.use_smpl_sdf =  False
    config.use_pred_depth = False
    config.use_smpl_depth = True
    # config.learning_rate /= 10.
    t = Trainer(use_hrnet=True)
    data_path=''
    checkpoint_save_path=''
    t.train_network(
        data_path,
        checkpoint_save_path,
        config.epoch_num,
        pre_trained_model=None,
        # previous_optimizer='checkpoints/rgbd_w_smpl_depth_normal_joints_only_sdf_v1',
        start_epoch=0
    )
    test_dir= ''
    result_dir = ''
    checkpoint_path = ''
    t.eval_network(test_dir,
                checkpoint_path,
                result_dir)


if __name__ == '__main__':
    start_time = time.time()
    main_train_geometry()
    end_time = time.time()
    print('total time: ', end_time - start_time)
