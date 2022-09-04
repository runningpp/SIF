import numpy as np
import cv2
import torch
import torchvision
import trimesh
from pytorch3d.io import load_obj
import os, sys
from termcolor import colored
import os.path as osp
from scipy.spatial import cKDTree
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from pytorch3d.structures import Meshes
from util.render_utils import Pytorch3dRasterizer
from pytorch3d.renderer.mesh import rasterize_meshes
# from psbody.mesh import Mesh
import scipy.io as sio

def barycentric_coordinates_of_projection(points, vertices) -> object:
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf

    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    # (p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 0], vertices[:, 0]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights

def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) *
                     nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def cal_sdf_batch(verts, faces,vis,cmaps, points):
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]

    normals = Meshes(verts, faces).verts_normals_padded()

    triangles = face_vertices(verts, faces)
    normals = face_vertices(normals, faces)
    vis = face_vertices(vis, faces)
    cmaps = face_vertices(cmaps, faces)

    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    # print(pts_ind)
    # print(residues.shape)
    closest_triangles = torch.gather(triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    closest_normals = torch.gather(normals, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    closest_vis = torch.gather(vis, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 1)).view(-1, 3, 1)
    closest_cmaps = torch.gather(cmaps, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    # print(closest_triangles.shape,point.shape)
    bary_weights = barycentric_coordinates_of_projection(points[0], closest_triangles)
    # print(bary_weights.shape)
    pts_vis = (closest_vis * bary_weights[:, :, None]).sum(1).unsqueeze(0).ge(1e-1)
    pts_norm = (closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0) #* torch.tensor([-1.0, 1.0, -1.0]).type_as(normals)
    pts_dist = torch.sqrt(residues) #/ torch.sqrt(torch.tensor(3))
    closet_points = (closest_triangles*bary_weights[:,:,None]).sum(1).unsqueeze(0)
    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
    pts_cmap = (closest_cmaps * bary_weights[:, :, None]).sum(1).unsqueeze(0)
    # print(closet_points)
    return pts_sdf, pts_norm,closet_points,pts_vis,pts_cmap

def get_visibility(xyz, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """
    # xyz *= torch.tensor([0,0,-1]).to(xyz.device)
    # xyz = torch.cat((xy, -z), dim=1)
    # xyz = (xyz + 1.0) / 2.0
    # faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    # print(torch.is_tensor(xzz),torch.is_tensor(faces))
    meshes_screen = Meshes(verts=xyz[None, ...], faces= faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(xyz.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask

def load_cams( data_fd, view_id):
    cam_data = sio.loadmat(os.path.join(data_fd, 'cams.mat'))
    cam_r = np.float32(cam_data['cam_rs'][view_id])
    cam_t = np.float32(cam_data['cam_ts'][view_id])
    center_depth = np.squeeze(cam_data['center_depth'])[view_id]
    center_depth = np.float32(center_depth)
    return cam_r, cam_t, center_depth


def cal_sdf_batch1(verts, faces, vis, points):
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]
    print(verts.shape,faces.shape,vis.shape)
    batch = verts.shape[0]
    normals = Meshes(verts, faces).verts_normals_padded()

    triangles = face_vertices(verts, faces)
    normals = face_vertices(normals, faces)

    vis = face_vertices(vis, faces)

    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    print(residues.shape)
    closest_triangles = torch.gather(triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    closest_normals = torch.gather(normals, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)

    closest_vis = torch.gather(vis, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 1)).view(-1, 3, 1)
    print(closest_triangles.shape, points.shape)
    bary_weights = barycentric_coordinates_of_projection(points.reshape(-1,3), closest_triangles)
    # bary_weights = barycentric_coordinates_of_projection(points[0], closest_triangles)
    print(bary_weights.shape)

    pts_vis = (closest_vis * bary_weights[:, :, None]).sum(1).unsqueeze(0).ge(1e-1).reshape(batch,-1,1)
    pts_norm = ((closest_normals * bary_weights[:, :, None]).sum(1).unsqueeze(0) * torch.tensor(
        [-1.0, 1.0, -1.0]).type_as(normals)).reshape(batch,-1,3)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
    print(pts_dist.shape,pts_signs.shape)
    print(pts_vis.shape,pts_norm.shape,pts_sdf.shape)
    return pts_sdf, pts_norm, pts_vis

def cal_vis_batch(verts, faces, vis, points):
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]


    triangles = face_vertices(verts, faces)

    vis = face_vertices(vis, faces)
    # print(points.shape,triangles.shape)
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    closest_triangles = torch.gather(triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)


    closest_vis = torch.gather(vis, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 1)).view(-1, 3, 1)

    bary_weights = barycentric_coordinates_of_projection(points[0], closest_triangles)


    pts_vis = (closest_vis * bary_weights[:, :, None]).sum(1).unsqueeze(0).ge(1e-1)


    return pts_vis


def cal_only_sdf_batch(verts, faces, points):
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    # cmaps [B, N_vert, 3]

    triangles = face_vertices(verts, faces)
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    pts_dist = torch.sqrt(residues)

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf

if __name__ == "__main__":
    import trimesh
    import trimesh.proximity
    import time
    import cv2 as cv
    from os.path import join,exists,split
    # points = torch.randn((5000,3))*0.1
    smpl_pth = '/home/lpp/data/Twindom_smpl_915/126111535847796-h/smpl_mesh_scaled.obj'
    # mesh = trimesh.load(smpl_pth)
    # closest, distance, traingle_id = trimesh.proximity.closest_point(mesh, points.numpy())
    # barycentric = trimesh.triangles.points_to_barycentric(mesh.triangles[traingle_id], closest)
    # normals = mesh.vertex_normals[mesh.faces[traingle_id]]
    # normals = trimesh.unitize(
    #     (normals * barycentric.reshape((-1, 3, 1))).sum(axis=1))
    # print(closest)
    # smpl = trimesh.load(smpl_pth)
    # face = smpl.faces
    # verts = smpl.vertices
    # distance=trimesh.proximity.signed_distance(smpl,points)
    # print(distance)
    #
    # print(face.shape,verts.shape)
    # print(face[:10])
    # print(verts[:10])
    # face = torch.LongTensor(face)
    # verts = torch.FloatTensor(verts)
    # tri = face_vertices(verts,face)
    # print(tri[:10])
    smpl = load_obj(smpl_pth,device='cuda')
    # print(smpl)
    face = smpl[1][0]
    verts = smpl[0]
    # print(face.shape, verts.shape)
    # print(face[:10])
    # print(verts[:10])
    # t0 = time.time()
    # dis,norm,_ = cal_sdf_batch(verts.unsqueeze(0), face.unsqueeze(0),points.unsqueeze(0))
    # tri = face_vertices(verts.unsqueeze(0), face.unsqueeze(0))
    # residues, pts_ind, _ = point_to_mesh_distance(points.unsqueeze(0), tri)
    # pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))
    # print(residues,pts_dist)
    # print(tri[pts_ind])
    # print(tri[:10])
    # print(smpl[1][1][:10])
    # print(time.time()-t0)
    # print(norm)

    pth = '/home/lpp/data/TwinDom_PerspectRandom_Noisy/126111535847796-h'
    file = os.path.join(pth, 'sample_points.npz')
    boundary_samples_npz = np.load(file)
    inside = boundary_samples_npz['pts_inside']
    outside = boundary_samples_npz['pts_outside']
    points = torch.FloatTensor(np.concatenate([inside, outside], axis=0))

    cam_r, cam_t, _ = load_cams(pth, 10)
    cam_R = cv.Rodrigues(cam_r)[0]
    cam_R,cam_t = torch.FloatTensor(cam_R).cuda(),torch.FloatTensor(cam_t).cuda()
    pts_ = (torch.matmul(verts.unsqueeze(-2), cam_R) + cam_t).squeeze(-2)
    # print(pts_.shape,face.shape)
    vis = get_visibility(pts_,torch.as_tensor( face).long())
    # print(vis.shape)
    points = torch.tile(points,dims=(2,1,1))
    verts = torch.tile(verts, dims=(2, 1, 1))
    face = torch.tile(face, dims=(2, 1, 1))
    vis = torch.tile(vis, dims=(2, 1, 1))
    pts_sdf = cal_only_sdf_batch(verts, face, points.cuda())
    # _, _, pts_vis = cal_sdf_batch1(verts, face, vis.cuda(), points.cuda())
    # _,_,pts_vis = cal_sdf_batch1(verts[None,...],face[None,...],vis[None,...].cuda(),points[None,...].cuda())
    # print(pts_vis.shape)
    # pts_vis = pts_vis[1].squeeze(-1)
    print(pts_sdf.shape)
    pts_sdf = pts_sdf[1].squeeze(-1)
    with open('lpp1.obj', 'w') as fp:
        for i in range(points.shape[1]):
            point = points[0,i,:]
            # ov = pts_ov[i]
            ov = pts_sdf[i]
            # fp.write('v %f %f %f %d %d %d\n' % (point[0], point[1], point[2], ov[0],ov[1],ov[2])
            # if (ov > 0.5):
            #     fp.write('v %f %f %f 0 1 0\n' % (point[0], point[1], point[2]))
            # else:
            #     fp.write('v %f %f %f 1 0 0\n' % (point[0], point[1], point[2]))

            if (ov > 0.):
                fp.write('v %f %f %f 0 %d 0\n' % (point[0], point[1], point[2],10*ov))
            else:
                fp.write('v %f %f %f %d 0 0\n' % (point[0], point[1], point[2],-10*ov))