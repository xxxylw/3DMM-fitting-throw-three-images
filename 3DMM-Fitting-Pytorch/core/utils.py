import pickle
import numpy as np
import os
import torch


def pad_bbox(bbox, img_wh, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1+padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(img_wh[0] - x1, size_bb)
    size_bb = min(img_wh[1] - y1, size_bb)

    return [x1, y1, x1+size_bb, y1+size_bb]


def mymkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lm_weights(device):
    w = torch.ones(68).to(device)
    w[28:31] = 10
    w[48:68] = 10
    norm_w = w / w.sum()
    return norm_w


def save_obj(path, v, f, c):
    with open(path, 'w') as file:
        for i in range(len(v)):
            file.write('v %f %f %f %f %f %f\n' %
                       (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()

def save_obj_uv(path,v,f,uv,mtl_path):
    with open(path, 'w') as file:
        file.write("mtllib "+mtl_path+'\n')
        file.write("o mesh.face3d\n")
        for i in range(len(v)):
            file.write('v %f %f %f\n' %
                       (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(len(uv)):
            file.write('vt %f %f\n' %
                       (uv[i, 0], uv[i, 1]))
        file.write("g mesh.face3d\n")
        file.write("usemtl face3d\n")
        for i in range(len(f)):
            file.write('f %d/%d %d/%d %d/%d\n' % (f[i, 0],f[i, 0],f[i, 1],f[i, 1], f[i, 2], f[i, 2]))

    file.close()
def save_mtl(path,img_path):
    with open(path, 'w') as file:
        file.write("newmtl face3d\n")
        file.write("map_Kd "+img_path+"\n")

    file.close()

import numpy as np

def spherical_uv_mapping(verts):
    """
    对一批顶点做简单的球面 UV 展开：
      u = (atan2(z, x) / (2π)) + 0.5
      v = acos(y / r) / π
    其中 r = √(x²+y²+z²)
    返回 [V,2] 数组，u,v 都落在 [0,1]
    """
    x, y, z = verts[:,0], verts[:,1], verts[:,2]
    r = np.linalg.norm(verts, axis=1)
    # 避免除零
    r[r==0] = 1e-8

    u = np.arctan2(z, x) / (2*np.pi) + 0.5
    v = np.arccos(y / r) / np.pi

    # 有的工具要求 v 从底向上，或反过来，根据你贴图习惯可取 v=1-v
    return np.stack([u, 1-v], axis=1)  # [V,2]
