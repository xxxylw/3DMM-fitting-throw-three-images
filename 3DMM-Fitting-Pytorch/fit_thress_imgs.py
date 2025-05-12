from facenet_pytorch import MTCNN
from core.options import VideoFittingOptions
from PIL import Image
import cv2
import face_alignment
import numpy as np
from core import get_recon_model
import os
import torch
import core.utils as utils
from tqdm import tqdm
import core.losses as losses
import shutil
import random
import glob
import pickle
from multiprocessing import Process, set_start_method
from core.fitting_dataset import FittingDataset
import argparse

def pad_to_square_center(image, size=600, pad_value=0):
    """
    将 image 填充到 size x size 的正方形，左右/上下居中补 pad_value。
    """
    h, w = image.shape[:2]
    if h > size or w > size:
        raise ValueError(f"输入图像 {w}x{h} 超过目标 {size}x{size}")

    pad_vert = size - h
    pad_horz = size - w
    top    = pad_vert // 2
    bottom = pad_vert - top
    left   = pad_horz // 2
    right  = pad_horz - left

    if image.ndim == 3:
        val = [pad_value] * image.shape[2]
    else:
        val = pad_value

    padded = cv2.copyMakeBorder(
        image,
        top, bottom,
        left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=val
    )
    return padded

def fit_coeffs(args, device, worker_ind):
    id_coeff = np.load(args.id_npy_path)
    tex_coeff = np.load(args.tex_npy_path)
    recon_model = get_recon_model(model=args.recon_model,
                                  device=device,
                                  batch_size=1,
                                  img_size=args.tar_size)
    recon_model.init_coeff_tensors(id_coeff=id_coeff, tex_coeff=tex_coeff)
    fitting_dataset = FittingDataset(args.lm_pkl_path)
    fitting_data_loader = torch.utils.data.DataLoader(dataset=fitting_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1,
                                                      drop_last=False)
    lm_weights = utils.get_lm_weights(device)
    res_dict = {}
    num_imgs = len(fitting_dataset)
    # store initial rotation from first frame
    initial_rot = None
    initial_trans = None

    for batch_ind, cur_batch in tqdm(enumerate(fitting_data_loader)):
        print('fitting %d/%d' % (batch_ind, num_imgs))
        # before fitting frame>0, reset rotation to initial
        if batch_ind > 0 and initial_rot is not None:
            recon_model.get_rot_tensor().data.copy_(initial_rot)
            recon_model.get_trans_tensor().data.copy_(initial_trans)

        lms, img, img_keys = cur_batch
        lms = lms.to(device)
        imgs = img.to(device)
        # rigid fitting...
        rigid_optimizer = torch.optim.Adam([recon_model.get_rot_tensor(),
                                            recon_model.get_trans_tensor()],
                                           lr=args.rf_lr)
        num_iters = args.first_rf_iters if batch_ind == 0 else args.rest_rf_iters
        for i in range(num_iters):
            rigid_optimizer.zero_grad()
            pred_dict = recon_model(
                recon_model.get_packed_tensors(), render=False)
            lm_loss_val = losses.lm_loss(
                pred_dict['lms_proj'], lms, lm_weights, img_size=args.tar_size)
            total_loss = args.lm_loss_w * lm_loss_val
            total_loss.backward()
            rigid_optimizer.step()
        print('done rigid fitting. lm_loss: %f' %
              lm_loss_val.detach().cpu().numpy())

        # after first frame fitting, record initial
        if batch_ind == 0:
            initial_rot = recon_model.get_rot_tensor().detach().clone()
            initial_trans = recon_model.get_trans_tensor().detach().clone()

        # non-rigid fitting...
        nonrigid_optimizer = torch.optim.Adam(
            [recon_model.get_id_tensor(), recon_model.get_exp_tensor(),
             recon_model.get_gamma_tensor(), recon_model.get_tex_tensor(),
             recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
            lr=args.nrf_lr)
        num_iters = args.first_nrf_iters if batch_ind == 0 else args.rest_nrf_iters
        for i in range(num_iters):
            nonrigid_optimizer.zero_grad()
            pred_dict = recon_model(
                recon_model.get_packed_tensors(), render=True)
            rendered_img = pred_dict['rendered_img']
            lms_proj = pred_dict['lms_proj']
            face_texture = pred_dict['face_texture']

            mask = rendered_img[:, :, :, 3].detach()
            photo_loss_val = losses.photo_loss(
                rendered_img[:, :, :, :3], imgs, mask > 0)
            lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights,
                                         img_size=args.tar_size)
            id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
            exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())
            tex_reg_loss = losses.get_l2(recon_model.get_tex_tensor())
            tex_loss_val = losses.reflectance_loss(
                face_texture, recon_model.get_skinmask())

            loss = lm_loss_val*args.lm_loss_w*0.5 + \
                id_reg_loss*args.id_reg_w + \
                exp_reg_loss*args.exp_reg_w + \
                tex_reg_loss*args.tex_reg_w + \
                tex_loss_val*args.tex_w*10 + \
                photo_loss_val*args.rgb_loss_w*10

            # regularizers for rotation and translation
            if batch_ind > 0:
                rot_diff = recon_model.get_rot_tensor() - initial_rot
                trans_diff = recon_model.get_trans_tensor() - initial_trans
                rot_reg = torch.square(rot_diff).sum()
                trans_reg = torch.square(trans_diff).sum()
                loss += rot_reg * args.rot_reg_w * 0.5
                loss += trans_reg * args.trans_reg_w * 0.5

            loss.backward()
            nonrigid_optimizer.step()

        # record results
        cur_k = img_keys[0]
        res_dict[cur_k] = {
            'rot': recon_model.get_rot_tensor().detach().cpu().numpy().reshape(1, -1),
            'trans': recon_model.get_trans_tensor().detach().cpu().numpy().reshape(1, -1),
            'gamma': recon_model.get_gamma_tensor().detach().cpu().numpy().reshape(1, -1),
            'exp': recon_model.get_exp_tensor().detach().cpu().numpy().reshape(1, -1)
        }

    output_dict = {'id': id_coeff, 'tex': tex_coeff, 'fitting_res': res_dict}
    with open(os.path.join(args.cache_folder, '%d_fitting.pkl' % worker_ind), 'wb') as f:
        pickle.dump(output_dict, f)


def fit_shape(args, device):
    recon_model = get_recon_model(model=args.recon_model,
                                  device=device,
                                  batch_size=args.nframes_shape,
                                  img_size=args.tar_size)
    fitting_dataset = FittingDataset(args.lm_pkl_path)
    fitting_data_loader = torch.utils.data.DataLoader(dataset=fitting_dataset,
                                                      batch_size=args.nframes_shape,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=False)
    for cur_batch in fitting_data_loader:
        lms, img, img_keys = cur_batch
        lms = lms.to(device)
        imgs = img.to(device)
        break # 只取第一帧
    lm_weights = utils.get_lm_weights(device)
    print('start rigid fitting in fit_shape')
    rigid_optimizer = torch.optim.Adam([recon_model.get_rot_tensor(),
                                        recon_model.get_trans_tensor()],
                                       lr=args.rf_lr)
    for i in tqdm(range(args.first_rf_iters)):
        rigid_optimizer.zero_grad()
        pred_dict = recon_model(recon_model.get_packed_tensors(), render=False)
        lm_loss_val = losses.lm_loss(
            pred_dict['lms_proj'], lms, lm_weights, img_size=args.tar_size)
        total_loss = args.lm_loss_w * lm_loss_val
        total_loss.backward()
        rigid_optimizer.step()
    print('done rigid fitting. lm_loss: %f' %
          lm_loss_val.detach().cpu().numpy())

    print('start non-rigid fitting')
    nonrigid_optimizer = torch.optim.Adam(
        [recon_model.get_id_tensor(), recon_model.get_exp_tensor(),
         recon_model.get_gamma_tensor(), recon_model.get_tex_tensor(),
         recon_model.get_rot_tensor(), recon_model.get_trans_tensor()],
        lr=args.nrf_lr)
    for i in tqdm(range(args.first_nrf_iters)):
        nonrigid_optimizer.zero_grad()

        pred_dict = recon_model(recon_model.get_packed_tensors(), render=True)
        rendered_img = pred_dict['rendered_img']
        lms_proj = pred_dict['lms_proj']
        face_texture = pred_dict['face_texture']

        mask = rendered_img[:, :, :, 3].detach()

        photo_loss_val = losses.photo_loss(
            rendered_img[:, :, :, :3], imgs, mask > 0)

        lm_loss_val = losses.lm_loss(lms_proj, lms, lm_weights,
                                     img_size=args.tar_size)
        id_reg_loss = losses.get_l2(recon_model.get_id_tensor())
        exp_reg_loss = losses.get_l2(recon_model.get_exp_tensor())
        tex_reg_loss = losses.get_l2(recon_model.get_tex_tensor())
        tex_loss_val = losses.reflectance_loss(
            face_texture, recon_model.get_skinmask())

        loss = lm_loss_val*args.lm_loss_w + \
            id_reg_loss*args.id_reg_w + \
            exp_reg_loss*args.exp_reg_w + \
            tex_reg_loss*args.tex_reg_w + \
            tex_loss_val*args.tex_w + \
            photo_loss_val*args.rgb_loss_w

        loss.backward()
        nonrigid_optimizer.step()

    loss_str = ''
    loss_str += 'lm_loss: %f\t' % lm_loss_val.detach().cpu().numpy()
    loss_str += 'photo_loss: %f\t' % photo_loss_val.detach().cpu().numpy()
    loss_str += 'tex_loss: %f\t' % tex_loss_val.detach().cpu().numpy()
    loss_str += 'id_reg_loss: %f\t' % id_reg_loss.detach().cpu().numpy()
    loss_str += 'exp_reg_loss: %f\t' % exp_reg_loss.detach().cpu().numpy()
    loss_str += 'tex_reg_loss: %f\t' % tex_reg_loss.detach().cpu().numpy()
    print('done non rigid fitting.', loss_str)

    np.save(args.id_npy_path, recon_model.get_id_tensor(
    ).detach().cpu().numpy().reshape(1, -1))
    np.save(args.tex_npy_path, recon_model.get_tex_tensor(
    ).detach().cpu().numpy().reshape(1, -1))

def process_imgs2(args, device):
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, flip_input=True, device=device)

    lm_list = []  # 存放 dict：{"path": ..., "lms": np.array[...]}

    for idx, img_p in enumerate(args.img_paths):
        img = cv2.imread(img_p)
        if img is None:
            raise FileNotFoundError(f"无法打开图像 {img_p}")
        orig_h, orig_w = img.shape[:2]

        # 裁剪 & resize 到 256×256
        face = img  # 这里不用裁bbox，直接整图
        face_resized = cv2.resize(face, (256, 256))

        # 提取 2D landmarks
        lms_all = fa.get_landmarks_from_image(face_resized)
        if lms_all is None:
            raise RuntimeError(f"第 {idx} 张图 landmark 提取失败")
        lms = lms_all[0][:, :2]  # (68,2)

        # 把 landmark 映射回原图坐标
        lms_orig = np.zeros_like(lms)
        lms_orig[:, 0] = lms[:, 0] * orig_w / 256
        lms_orig[:, 1] = lms[:, 1] * orig_h / 256

        # 保存一条记录
        lm_list.append({
            "path": os.path.abspath(img_p),
            "lms": lms_orig.astype(np.float32)
        })

        # （可选）可视化检查
        for (x, y) in lms_orig:
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(args.tmp_frame_folder, f"{idx}.png"), img)

        print(f"[{idx+1}/3] {img_p} 处理完成，landmarks 个数：{lms_orig.shape[0]}")

    # 存到 pkl
    with open(args.lm_pkl_path, "wb") as f:
        pickle.dump(lm_list, f)

    # 构造 v_info，fps 随便设，bbox 整图
    first_img = cv2.imread(args.img_paths[0])
    orig_h, orig_w = first_img.shape[:2]
    v_info = {
        'fps': 1,
        'frame_w': orig_w,
        'frame_h': orig_h,
        'bbox': (0, 0, orig_w, orig_h)
    }
    with open(args.v_info_path, "wb") as f:
        pickle.dump(v_info, f)

    print("静态图处理完成（无检测），共处理 %d 张图" % len(args.img_paths))



def merge_dict(args):

    final_dict = {}
    for i in range(args.nworkers):
        with open(os.path.join(args.cache_folder, '%d_fitting.pkl' % i), 'rb') as f:
            tmp_dict = pickle.load(f)
        if i == 0:
            final_dict['id'] = tmp_dict['id']
            final_dict['tex'] = tmp_dict['tex']
            final_dict['fitting_res'] = {}
        final_dict['fitting_res'].update(tmp_dict['fitting_res'])
    with open(args.fitting_pkl_path, 'wb') as f:
        pickle.dump(final_dict, f)


if __name__ == '__main__':
    args = VideoFittingOptions()
    args = args.parse()
    args.devices = ['cuda:%d' % i for i in range(args.ngpus)]
    # to avoid mult-processing runtime error
    set_start_method('spawn')
    parser = argparse.ArgumentParser(description="parameters of fitting")

    # remove cache files and create new folders
    if os.path.exists(args.cache_folder):
        shutil.rmtree(args.cache_folder)

    args.dataset_cache_folder = os.path.join('dataset', 'cache')
    args.tmp_face_folder = os.path.join(args.cache_folder, 'faces')
    args.basedir = os.path.join(args.dataset_cache_folder, args.id)
    args.face_undistorted = os.path.join(args.basedir, 'undistorted')
    args.face_padding = os.path.join(args.basedir, 'padded')
    args.tmp_frame_folder = os.path.join(args.cache_folder, 'frames')
    args.lm_pkl_path = os.path.join(args.cache_folder, 'lms.pkl')
    args.id_npy_path = os.path.join(args.cache_folder, 'id.npy')
    args.tex_npy_path = os.path.join(args.cache_folder, 'tex.npy')
    args.v_info_path = os.path.join(args.cache_folder, 'v_info.pkl')
    args.fitting_pkl_path = os.path.join(
        args.res_folder, os.path.basename(args.id)+'_fitting_res.pkl')
    args.out_video_path = os.path.join(
        args.res_folder, os.path.basename(args.id)[:-4]+'_recon_video.avi')

    utils.mymkdirs(args.cache_folder)
    # utils.mymkdirs(args.face_non_resized)
    utils.mymkdirs(args.tmp_face_folder)
    utils.mymkdirs(args.res_folder)
    utils.mymkdirs(args.tmp_frame_folder)

    undistort_path = args.face_undistorted
    img_extensions = ('*.jpg', '*.png')
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(args.face_padding, ext)))
    print(f'Found {len(img_paths)} images before ordering')
    basename = lambda p: os.path.basename(p).lower()
    mid_paths   = [p for p in img_paths if 'mid' in basename(p)]
    other_paths = [p for p in img_paths if 'mid' not in basename(p)]
    args.img_paths = mid_paths + other_paths
    print(f'args.img_paths[0]: {args.img_paths[0]}')
    process_imgs2(args, args.devices[0])

    fit_shape(args, args.devices[0])

    # fit frames using Process
    processes = []
    for i in range(args.nworkers):
        p = Process(target=fit_coeffs, args=(
            args, args.devices[i % args.ngpus], i))
        p.start()
        processes.append(p)

    for cur_p in processes:
        cur_p.join()
    # merge dict
    merge_dict(args)
