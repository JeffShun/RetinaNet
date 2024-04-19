import argparse
import os
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm_path', type=str, default='./train_data/dcms')
    parser.add_argument('--png_path', type=str, default='./train_data/pngs')
    args = parser.parse_args()
    return args

def normlize(img, win_clip=[1500,3500]):
    v_min, v_max = win_clip[0]-0.5*win_clip[1], win_clip[0]+0.5*win_clip[1]
    img = np.clip(img, v_min, v_max)
    img_norm = (img - img.min())/(img.max() - img.min())
    return img_norm


if __name__ == '__main__':
    args = parse_args()
    dcm_path = args.dcm_path
    png_path = args.png_path
    os.makedirs(png_path, exist_ok=True)
    for sample in tqdm(os.listdir(dcm_path)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dcm_path, sample)))[0]
        img = normlize(img)
        img = img*255
        img = Image.fromarray(img.astype("uint8"))
        img.save(png_path + "/" + sample.replace(".dcm",".png"))