
import numpy as np
from torch.utils import data
import torch
import os
import pickle
import cv2


class FittingDataset(torch.utils.data.Dataset):
    def __init__(self, lm_pkl_path, worker_num=1, worker_ind=0):
        super().__init__()
        with open(lm_pkl_path, 'rb') as f:
            self.lm_list = pickle.load(f)

        num = len(self.lm_list)
        per = num // worker_num
        self.start  = worker_ind * per
        self.length = num - self.start if worker_ind == worker_num - 1 else per

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rec      = self.lm_list[self.start + idx]
        img_path = rec["path"]
        lms      = rec["lms"]  # (68,2) numpy

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        img = img_bgr[:, :, ::-1].astype(np.float32)

        # 从文件名中判断视角
        basename = os.path.splitext(os.path.basename(img_path))[0].lower()
        if 'left' in basename:
            key = 'left'
        elif 'mid' in basename:
            key = 'mid'
        elif 'right' in basename:
            key = 'right'
        else:
            raise ValueError(f"Filename '{basename}' does not contain 'left','mid' or 'right'")

        lms_t = torch.from_numpy(lms)               # (68,2)
        # img_t = torch.from_numpy(img).permute(2,0,1) # (3,H,W)
        img_t = torch.from_numpy(img)

        return lms_t, img_t, key

