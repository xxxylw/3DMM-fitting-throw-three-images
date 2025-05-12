#!/usr/bin/env python3
'''
run this script to run the whole pipeline
example:
    python run_pipeline.py --id 1101
    you have to get the images in 
        dataset/
        └── 1101/
            ├── mid/
            │   ├── img1.jpg
            │   ├── img2.png
            │   └── ...
            ├── left/
            │   ├── img1.jpg
            │   ├── img2.png
            │   └── ...
            └── right/
                ├── img1.jpg
                ├── img2.png
                └── ...
    and the results wiil be saved in
        results/
        └── 1101/
            ├── xxx.obj
            ├── xxx.mtl 
            ├── xxx_uv.jpg

'''
import argparse
import subprocess
import os
import sys

def run_cmd(cmd, cwd=None):
    print(f"\n>>> Running: {' '.join(cmd)} (cwd={cwd or os.getcwd()})")
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    parser = argparse.ArgumentParser(description="自动化 3DMM pipeline")
    parser.add_argument("--id", required=True, help="要处理的 ID，例如 1101")
    parser.add_argument("--res-folder", default="results",
                        help="结果输出文件夹 (默认: results)")
    args = parser.parse_args()
    os.makedirs(args.res_folder, exist_ok=True)
    run_cmd(
        ["python", "undistort_image.py", "--id", args.id],
        cwd="scene"
    )
    run_cmd(
        ["python", "fit_thress_imgs.py",
         "--id", args.id,
         "--res_folder", args.res_folder]
    )
    run_cmd(
        ["python", "visualize2.py",
         "--id", args.id,
         "--res_folder", args.res_folder]
    )

    print(f"\n>>> 全部完成，结果保存在目录：{args.res_folder}")

if __name__ == "__main__":
    main()
