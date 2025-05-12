import cv2
import numpy as np
import os
import argparse
import cv2
import numpy as np
import os
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

def resize_to_max_side(image, max_side):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def undistort_and_save(input_path, out_undistort, out_padded,
                       camera_matrix, dist_coeffs, max_side=600):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像 {input_path}")
    h, w = img.shape[:2]

    # 1) 计算 new_cam
    new_cam, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1.0
    )

    # 2) 构建 map
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_cam, (w, h), cv2.CV_32FC1
    )

    # 3) remap
    remapped = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 4) 只缩放
    undistorted = resize_to_max_side(remapped, max_side)
    cv2.imwrite(out_undistort, undistorted)

    # 5) 再居中填充到正方形
    padded = pad_to_square_center(undistorted, max_side, pad_value=0)
    cv2.imwrite(out_padded, padded)

def process_images_for_camera(cams, img_folder, output_undistort, output_padded, camera_type):
    camera_matrix = cams[camera_type]["K"]
    dist_coeffs   = cams[camera_type]["dist"]

    os.makedirs(output_undistort, exist_ok=True)
    os.makedirs(output_padded, exist_ok=True)

    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png'))]
    for img_file in img_files:
        inp = os.path.join(img_folder, img_file)
        out_u = os.path.join(output_undistort, f"undistorted_{camera_type}_{img_file}")
        out_p = os.path.join(output_padded,     f"padded_{camera_type}_{img_file}")
        undistort_and_save(inp, out_u, out_p, camera_matrix, dist_coeffs, max_side=600)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default="../dataset")
    parser.add_argument('--output', type=str, default="../dataset/cache")
    parser.add_argument('--id', type=str, default='')
    args = parser.parse_args()

    base_img_folder = os.path.join(args.folder, args.id)
    if not os.path.exists(base_img_folder):
        print(f"不存在: {base_img_folder}")
        exit(1)

    # 准备输出路径
    output_base = os.path.join(args.output, args.id)
    
    undistort_dir = os.path.join(output_base, "undistorted")
    padded_dir    = os.path.join(output_base, "padded")

    # 加载标定
    calib = np.load('calibration_results_pri.npz')
    cams = {
        "right": {"K": calib["K_left"],  "dist": calib["dist_left"]},
        "mid":   {"K": calib["K_mid"],   "dist": calib["dist_mid"]},
        "left":  {"K": calib["K_right"], "dist": calib["dist_right"]},
    }

    for camera_type in ["mid", "right", "left"]:
        img_folder = os.path.join(base_img_folder, camera_type)
        process_images_for_camera(cams, img_folder,
                                  undistort_dir, padded_dir,
                                  camera_type)
    print("全部处理畸变完成。")
