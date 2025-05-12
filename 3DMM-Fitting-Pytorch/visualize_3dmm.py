import numpy as np
import cv2
import torch
import pickle
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation, Slerp
from core.options import VideoFittingOptions
from core import get_recon_model

# ---------------- Utility Functions ----------------

def rodrigues_to_quat(rvec):
    """
    Rodrigues 向量 -> 四元数 [x, y, z, w]
    """
    r = rvec.detach().cpu().numpy().reshape(3,)
    R_mat, _ = cv2.Rodrigues(r)
    return Rotation.from_matrix(R_mat).as_quat()


def quat_to_rodrigues(quat, device):
    """
    四元数 -> Rodrigues 向量
    """
    R_mat = Rotation.from_quat(quat).as_matrix()
    rvec, _ = cv2.Rodrigues(R_mat)
    return torch.tensor(rvec.reshape(3,), dtype=torch.float32, device=device)


def slerp_quat(q1, q2, alpha):
    """在两个四元数之间做球面线性插值"""
    times = [0, 1]
    rots = Rotation.from_quat([q1, q2])
    slerp = Slerp(times, rots)
    return slerp([alpha]).as_quat()[0]

# ---------------- Main Interpolation ----------------

def gen_interp_3dmm_video(args, device, num_interp=50):
    # 加载拟合结果
    with open(args.fitting_pkl_path, 'rb') as f:
        fit = pickle.load(f)
    # 全局不变系数 from k0
    id_t = torch.tensor(fit['id'], dtype=torch.float32, device=device)
    tex_t= torch.tensor(fit['tex'],dtype=torch.float32, device=device)
    res = fit['fitting_res']

    model = get_recon_model(
        model=args.recon_model,
        device=device,
        batch_size=1,
        img_size=args.tar_size)

    # 关键帧参数
    k0, k1, k2 = res[0], res[1], res[2]

        # 保持 exp, trans, gamma 为 k0 (already shape [1, dim])
    exp0   = torch.tensor(k0['exp'],  dtype=torch.float32, device=device)  # [1, exp_dim]
    trans0 = torch.tensor(k0['trans'],dtype=torch.float32, device=device)  # [1, 3]
    gamma0 = torch.tensor(k0['gamma'],dtype=torch.float32, device=device)  # [1, gamma_dim]

    # 旋转四元数
    q0 = rodrigues_to_quat(torch.tensor(k0['rot'], dtype=torch.float32, device=device).squeeze())
    q1 = rodrigues_to_quat(torch.tensor(k1['rot'], dtype=torch.float32, device=device).squeeze())
    q2 = rodrigues_to_quat(torch.tensor(k2['rot'], dtype=torch.float32, device=device).squeeze())
    q0 = rodrigues_to_quat(torch.tensor(k0['rot'], dtype=torch.float32, device=device).squeeze())
    q1 = rodrigues_to_quat(torch.tensor(k1['rot'], dtype=torch.float32, device=device).squeeze())
    q2 = rodrigues_to_quat(torch.tensor(k2['rot'], dtype=torch.float32, device=device).squeeze())

    # 视频 writer 使用 mp4v
    size = (args.tar_size, args.tar_size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(args.out_video_path, fourcc, 25, size)

    def render(rot_vec):
        rot = rot_vec.unsqueeze(0)  # [1,3]
        coeffs = model.merge_coeffs(id_t, exp0, tex_t, rot, gamma0, trans0)
        pd = model(coeffs, render=True)['rendered_img'].cpu().numpy().squeeze()
        return pd[:, :, :3].astype(np.uint8)

    alphas = np.linspace(0, 1, num_interp + 1)

    # 从 k1 -> k0 -> k2
    for alpha in alphas:
        q_i = slerp_quat(q1, q0, alpha)
        r_vec = quat_to_rodrigues(q_i, device)
        fr = render(r_vec)
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    for alpha in alphas:
        q_i = slerp_quat(q0, q2, alpha)
        r_vec = quat_to_rodrigues(q_i, device)
        fr = render(r_vec)
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    vw.release()
    print(f"Interpolated 3DMM video saved to {args.out_video_path}")

if __name__ == '__main__':
    args = VideoFittingOptions().parse()
    args.devices = ['cuda:%d' % i for i in range(args.ngpus)]
    # 更新路径为 mp4
    args.fitting_pkl_path = os.path.join(
        args.res_folder,
        os.path.basename(args.v_path)[:-4] + '_fitting_res.pkl')
    args.out_video_path = os.path.join(
        args.res_folder,
        os.path.basename(args.v_path)[:-4] + '_interp.mp4')

    gen_interp_3dmm_video(args, args.devices[0], num_interp=50)
