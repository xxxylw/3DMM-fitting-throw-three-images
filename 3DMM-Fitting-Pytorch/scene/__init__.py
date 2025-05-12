import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov

class SceneMultiViewFromCalibration:
    def __init__(self, datadir, calib_npz_path, white_background, device):
        print("in SceneMultiViewFromCalibration")
        self.bg_image = torch.zeros((3, 600, 450))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        self.cameras = []

        calib = np.load(calib_npz_path)
        R_mid = np.eye(3)
        T_mid = np.zeros((3,))

        # 取出标定给你的 mid<–>left、mid<–>right 变换
        R_ml = calib["R_ml"]          # OpenCV 给出的旋转（mid→left 或 left→mid，取决于你当时怎么调用）
        T_ml = calib["T_ml"].flatten()
        R_mr = calib["R_mr"]
        T_mr = calib["T_mr"].flatten()

        # 反向一下，得到 world(mid)→left/right
        R_left  = R_ml.T
        T_left  = - R_ml.T @ T_ml

        R_right = R_mr.T
        T_right = - R_mr.T @ T_mr


        # 获取三个视角的参数
        cams = {
            "left": {"K": calib["K_left"], "dist": calib["dist_left"], "R": calib["R_ml"], "T": calib["T_ml"].flatten()},
            "mid": {"K": calib["K_mid"], "dist": calib["dist_mid"], "R": np.eye(3), "T": np.zeros((3,))},
            "right": {"K": calib["K_right"], "dist": calib["dist_right"], "R": calib["R_mr"], "T": calib["T_mr"].flatten()}
        }

        for view_name, params in cams.items():
            K = params["K"]
            fl_x, fl_y = K[0, 0], K[1, 1]
            orig_w, orig_h = 450, 600  # 原始图像分辨率，如有 resize 请调整
            FovY = focal2fov(fl_y, orig_h)
            FovX = focal2fov(fl_x, orig_w)

            R = params["R"].T
            T = params["T"]

            image_dir = os.path.join(datadir, "imgs", view_name)
            parsing_dir = os.path.join(datadir, "parsing", view_name)

            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

            for i, fn in enumerate(image_files):
                image_name = os.path.splitext(fn)[0]
                image_path = os.path.join(image_dir, fn)
                image = Image.open(image_path)
                resized_image_rgb = PILtoTensor(image)
                gt_image = resized_image_rgb[:3, ...]


                # # alpha mask
                # alpha_path = os.path.join(parsing_dir, image_name + "_alpha.png")
                # if os.path.exists(alpha_path):
                #     alpha = PILtoTensor(Image.open(alpha_path))
                #     gt_image = gt_image * alpha + self.bg_image * (1 - alpha)

                # head mask
                head_mask_path = os.path.join(parsing_dir, image_name + ".jpg")
                assert os.path.exists(head_mask_path), f"Missing head mask: {head_mask_path}"
                head_mask = PILtoTensor(Image.open(head_mask_path))
                # print(f"head mask shape: {head_mask.shape}")
                # print(f"gt_image shape: {gt_image.shape}")
                # print(f"imagepath: {image_path}")
                # print(f"head_mask_path: {head_mask_path}")
                gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

                cam = Camera(
                    colmap_id=len(self.cameras), R=R, T=T,
                    FoVx=FovX, FoVy=FovY,
                    image=gt_image, head_mask=head_mask,
                    image_name=image_name, uid=len(self.cameras),
                    data_device=device
                )
                self.cameras.append(cam)
        print("finish SceneMultiViewFromCalibration")

    def getCameras(self):
        return self.cameras

