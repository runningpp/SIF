# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Runtime Configuration"""
from __future__ import print_function, absolute_import, division
import numpy as np
import torch

device = torch.device("cuda")
img_w = 512
img_h = 512
cam_f = 550

################
# training paras

subject_number = 300
epoch_num = 2000
batch_size = 6
view_num = 1
# view_num = 4
render_view_number = 60

useRFF = False
color_useRFF = False
#############attention experiments################################
use_Attn = False
add_mlp = True
use_position=False
##################################################

learning_rate = 2e-4
clr_learning_rate = 2e-4

input_mode = 'rgbd'
# input_mode = 'rgb_only'
# input_mode = 'depth_mask'


num_out = 256

psdf_zero_band_width = 0.01

erode_iterations = 2
boundary_cut = True

###############
# testing paras
testing_res = 256
infer_color = True
save_input_rgbd = False
point_group_size = 256 * 32
# point_group_size = 128 * 32

######
img_feat = 64

work_num=10

use_cutoff = False
cut_dist = 500*0.00035
use_joints_sample = False
use_smpl_senmantic = False
only_use_smpl = False
only_use_smpl_sign = False
use_smpl_depth = False
use_smpl_sdf = False
use_pred_depth=False
use_vis=False
add_z = False