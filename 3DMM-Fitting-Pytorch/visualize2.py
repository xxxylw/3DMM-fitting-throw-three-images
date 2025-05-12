import numpy as np
import cv2
import torch
import glob
import pickle
import os
from tqdm import tqdm
import core.utils as utils
from scipy.spatial.transform import Rotation, Slerp
from core.options import VideoFittingOptions
from core import get_recon_model

# ---------------- Main Interpolation ----------------

def gen_obj_with_tex(args, device):

    first_img = cv2.imread(args.img_paths[0])
    basename = os.path.basename(args.img_paths[0])
    
    # 加载拟合结果
    with open(args.fitting_pkl_path, 'rb') as f:
        fit = pickle.load(f)
    # 全局不变系数 from k0
    id_t = torch.tensor(fit['id'], dtype=torch.float32, device=device)
    tex_t= torch.tensor(fit['tex'],dtype=torch.float32, device=device)
    res = fit['fitting_res']

    recon_model = get_recon_model(
        model=args.recon_model,
        device=device,
        batch_size=1,
        img_size=args.tar_size)

    # 关键帧参数
    # k0, k1, k2 = res[0], res[1], res[2]
    print("Available keys in fitting_res:", list(res.keys()))
    k0 = res['mid']

        # 保持 exp, trans, gamma 为 k0 (already shape [1, dim])
    exp0   = torch.tensor(k0['exp'],  dtype=torch.float32, device=device)  # [1, exp_dim]
    trans0 = torch.tensor(k0['trans'],dtype=torch.float32, device=device)  # [1, 3]
    gamma0 = torch.tensor(k0['gamma'],dtype=torch.float32, device=device)  # [1, gamma_dim]
    rot0   = torch.tensor(k0['rot'],  dtype=torch.float32, device=device)  # [1, 3]

    pred_dict = recon_model(recon_model.merge_coeffs(
            id_t, exp0, tex_t,
            rot0, gamma0, trans0), render=True)
    
    vs_uv = recon_model.project_vs(pred_dict['vs']).cpu().numpy().squeeze()
    uv = vs_uv / 600
    vs = pred_dict['vs'].cpu().numpy().squeeze()
    tri = pred_dict['tri'].cpu().numpy().squeeze()

    out_obj_dir = os.path.join(args.res_folder, args.id)
    os.makedirs(out_obj_dir, exist_ok=True)
    out_obj_path = os.path.join(out_obj_dir, basename+'_mesh.obj')
    undistort_path = os.path.join(args.dataset_cache_folder, args.id, 'undistorted')
    # mid_img_path = None
    # for fname in os.listdir(undistort_path):
    #     if 'mid' in fname.lower():  # 忽略大小写匹配
    #         mid_img_path = os.path.join(undistort_path, fname)
    #         break

    # if mid_img_path is None:
    #     raise FileNotFoundError(f"No file containing 'mid' found in {undistort_path}")
    # img = cv2.imread(mid_img_path)
    # if img is None:
    #     raise ValueError(f"Failed to read image: {mid_img_path}")

    # tex_path = os.path.join(out_obj_dir, f"{args.id}_uv.jpg")
    # cv2.imwrite(tex_path, img)
    uv_path = os.path.join(out_obj_dir, basename+'_uv.jpg')
    cv2.imwrite(uv_path, first_img)
    mtl_path = basename + '_uv.mtl'
    utils.save_mtl(os.path.join(out_obj_dir, mtl_path),uv_path)
    utils.save_obj_uv(out_obj_path, vs, tri + 1, uv, mtl_path)
    print(f'have saved mesh to {out_obj_path}')

if __name__ == '__main__':
    args = VideoFittingOptions().parse()
    args.devices = ['cuda:%d' % i for i in range(args.ngpus)]
    # 更新路径为 mp4
    args.fitting_pkl_path = os.path.join(
        args.res_folder,
        os.path.basename(args.id) + '_fitting_res.pkl')
    args.dataset_cache_folder = os.path.join('dataset', 'cache')
    undistort_path = os.path.join(args.dataset_cache_folder, args.id, 'undistorted')
    padding_path = os.path.join(args.dataset_cache_folder, args.id, 'padded')
    img_extensions = ('*.jpg', '*.png')
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(padding_path, ext)))
    print(f'Found {len(img_paths)} images before ordering')
    basename = lambda p: os.path.basename(p).lower()
    mid_paths   = [p for p in img_paths if 'mid' in basename(p)]
    other_paths = [p for p in img_paths if 'mid' not in basename(p)]
    args.img_paths = mid_paths + other_paths
    
    # args.img_paths = ["./data/1005_mid_noresied.png", "./data/1005_left_noresized.png", "./data/1005_right_noresized.png"]

    gen_obj_with_tex(args, args.devices[0])
