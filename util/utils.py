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

"""Utilization"""

from __future__ import print_function, absolute_import, division
import numpy as np
import scipy
import math
import os
import sys
import cv2 as cv
import datetime
from subprocess import call

def read_smpl_constants(folder):
    """Load smpl vertex code"""
    smpl_vtx_std = np.loadtxt(os.path.join(folder, 'vertices.txt'))
    min_x = np.min(smpl_vtx_std[:, 0])
    max_x = np.max(smpl_vtx_std[:, 0])
    min_y = np.min(smpl_vtx_std[:, 1])
    max_y = np.max(smpl_vtx_std[:, 1])
    min_z = np.min(smpl_vtx_std[:, 2])
    max_z = np.max(smpl_vtx_std[:, 2])

    smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
    smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
    smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
    smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))

    """Load smpl faces & tetrahedrons"""
    smpl_faces = np.loadtxt(os.path.join(folder, 'faces.txt'), dtype=np.int32) - 1
    smpl_face_code = (smpl_vertex_code[smpl_faces[:, 0]] +
                      smpl_vertex_code[smpl_faces[:, 1]] + smpl_vertex_code[smpl_faces[:, 2]]) / 3.0
    smpl_tetras = np.loadtxt(os.path.join(folder, 'tetrahedrons.txt'), dtype=np.int32) - 1
    return smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras

class Logger(object):
    """Console logger"""

    def __init__(self):
        self.file = None
        self.buffer = ''

    def set_log_file(self, filename):
        """sets the log file"""
        assert self.file is None
        self.file = open(filename, 'wt')
        if self.buffer is not None:
            self.file.write(self.buffer)
            self.buffer = None

    def write(self, *args):
        """writes output message"""
        now = datetime.datetime.now()
        dtstr = now.strftime('%Y-%m-%d %H:%M:%S')
        t_msg = '[%s]' % dtstr + ' %s' % ' '.join(map(str, args))

        print(t_msg)
        if self.file is not None:
            self.file.write(t_msg + '\n')
        else:
            self.buffer += t_msg

    def flush(self):
        if self.file is not None:
            self.file.flush()


logger = Logger()


def safe_mkdir(dir):
    """Performs mkdir after checking existence"""
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        logger.write('WARNING: %s already exists. ' % dir)


def create_folders(path):
    fd_trees = path.split('/')
    p = ''
    for i in range(len(fd_trees)):
        p = os.path.join(p, fd_trees[i])
        if not os.path.exists(p):
            os.mkdir(p)


def get_subfolder_list(dir):
    return os.listdir(dir)


def get_file_list(dir, pattern='*.*'):
    import glob
    return glob.glob(os.path.join(dir, pattern))


class ProgressBar(object):
    """Progress bar displaying in stdout"""

    def __init__(self, width=40):
        self._w = width
        self._total = 1
        self._curr = 0
        self._curr_str = ''

    def start(self, total_count):
        """creates a progress bar"""
        self._total = total_count
        self._curr = 0
        self._curr_str = "[%s%s] (%d/%d)" % ('', ' ' * self._w,
                                             self._curr, self._total)
        sys.stdout.write(self._curr_str)
        sys.stdout.flush()

    def count(self, c=1):
        """updates the progress"""
        # remove previous output
        sys.stdout.write("\b" * len(self._curr_str))
        sys.stdout.flush()

        # update output
        self._curr = self._curr + c
        step = int(self._w * self._curr / self._total)
        self._curr_str = "[%s%s] (%d/%d)" % ('#' * step, ' ' * (self._w - step),
                                             self._curr, self._total)
        sys.stdout.write(self._curr_str)
        sys.stdout.flush()

    def end(self):
        """finishes the job and stops the progress bar"""
        sys.stdout.write('\nFinished. \n')
        sys.stdout.flush()
        self._total = 1
        self._curr = 0
        self._curr_str = ''


# mesh pre-processing
# =====================================================
def calc_normal(mesh):
    """calculates surface normal"""
    from opendr.lighting import VertNormals
    n = VertNormals(f=mesh['f'], v=mesh['v'])
    return n.r


def flip_axis_in_place(mesh, x_sign, y_sign, z_sign):
    """flips model along some axes"""
    mesh['v'][:, 0] *= x_sign
    mesh['v'][:, 1] *= y_sign
    mesh['v'][:, 2] *= z_sign

    if mesh['vn'] is not None and len(mesh['vn'].shape) == 2:
        mesh['vn'][:, 0] *= x_sign
        mesh['vn'][:, 1] *= y_sign
        mesh['vn'][:, 2] *= z_sign
    return mesh


def transform_mesh_in_place(mesh, trans, scale):
    """
    Transforms mesh
    Note that it will perform translation first, followed by scaling
    Also note that the transformation happens in-place
    """
    mesh['v'][:, 0] += trans[0]
    mesh['v'][:, 1] += trans[1]
    mesh['v'][:, 2] += trans[2]

    mesh['v'] *= scale
    return mesh


def rotate_model_in_place_xyz(mesh, x_r, y_r, z_r):
    """rotates model (x-axis first, then y-axis, and then z-axis)"""
    mat_x, _ = cv.Rodrigues(np.asarray([x_r, 0, 0], dtype=np.float32))
    mat_y, _ = cv.Rodrigues(np.asarray([0, y_r, 0], dtype=np.float32))
    mat_z, _ = cv.Rodrigues(np.asarray([0, 0, z_r], dtype=np.float32))
    mat = np.matmul(np.matmul(mat_x, mat_y), mat_z)

    v = mesh['v'].transpose()
    v = np.matmul(mat, v)
    mesh['v'] = v.transpose()

    if 'vn' in mesh and mesh['vn'] is not None and len(mesh['vn'].shape) == 2:
        n = mesh['vn'].transpose()
        n = np.matmul(mat, n)
        mesh['vn'] = n.transpose()

    return mesh


def rotate_model_in_place_zyx(mesh, x_r, y_r, z_r):
    """rotates model (x-axis first, then y-axis, and then z-axis)"""
    mat_x, _ = cv.Rodrigues(np.asarray([x_r, 0, 0], dtype=np.float32))
    mat_y, _ = cv.Rodrigues(np.asarray([0, y_r, 0], dtype=np.float32))
    mat_z, _ = cv.Rodrigues(np.asarray([0, 0, z_r], dtype=np.float32))
    mat = np.matmul(np.matmul(mat_z, mat_y), mat_x)

    v = mesh['v'].transpose()
    v = np.matmul(mat, v)
    mesh['v'] = v.transpose()

    if 'vn' in mesh and mesh['vn'] is not None and len(mesh['vn'].shape) == 2:
        n = mesh['vn'].transpose()
        n = np.matmul(mat, n)
        mesh['vn'] = n.transpose()

    return mesh


def calc_transform_params(mesh, smpl, hb_ratio=1.0, scale_noise=0):
    """
    Calculates the transformation params used to transform the mesh to unit
    bounding box centered at the origin. Returns translation and scale.
    Note that to use the returned parameters, you should perform translation
    first, followed by scaling
    """
    min_x = np.min(mesh['v'][:, 0])
    max_x = np.max(mesh['v'][:, 0])
    min_y = np.min(mesh['v'][:, 1])
    max_y = np.max(mesh['v'][:, 1])
    min_z = np.min(mesh['v'][:, 2])
    max_z = np.max(mesh['v'][:, 2])

    min_x = min(np.min(smpl['v'][:, 0]), min_x)
    max_x = max(np.max(smpl['v'][:, 0]), max_x)
    min_y = min(np.min(smpl['v'][:, 1]), min_y)
    max_y = max(np.max(smpl['v'][:, 1]), max_y)
    min_z = min(np.min(smpl['v'][:, 2]), min_z)
    max_z = max(np.max(smpl['v'][:, 2]), max_z)

    trans = -np.array([(min_x + max_x) / 2, (min_y + max_y) / 2,
                       (min_z + max_z) / 2])

    scale_inv = max(max((max_x - min_x) / hb_ratio, (max_y - min_y)),
                    (max_z - min_z) / hb_ratio)
    scale_inv *= (1.05 + scale_noise)
    scale_inv += 1e-3  # avoid division by zero
    scale = 1.0 / scale_inv
    return trans, scale
