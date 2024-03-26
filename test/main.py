import argparse
import glob
import os
import sys
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.font_manager as fm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import PredictModel, Predictor
from scipy.ndimage import zoom
import matplotlib.pylab as plt
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Test Object Detection')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='./data/input', type=str)
    parser.add_argument('--output_path', default='./data/output', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/Resnet50/90.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def inference(predictor: Predictor, img: np.ndarray):
    pred_array = predictor.predict(img)
    return pred_array


def parse_label(label_map, label_path):
    with open(label_path) as f:
        label_data = json.load(f)
        box_label = []
        for shape in label_data["shapes"]: 
            points = shape["points"]
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])

            label = shape["label"]
            label_map_rev = dict(zip(label_map.values(), label_map.keys()))
            box_label.append((x1, y1, x2, y2, label_map_rev[label]))
    return np.array(box_label) 


def save_result(img, preds, lable_map, save_path):
    img = (img-img.min())/(img.max()-img.min())
    img = (img * 255).astype("uint8")
    draw_img = np.zeros([img.shape[0], img.shape[1], 3])
    draw_img[:, :, 0] = img
    draw_img[:, :, 1] = img
    draw_img[:, :, 2] = img
    for pred in preds:
        cv2.rectangle(
            draw_img,
            (int(pred[0]), int(pred[1])),
            (int(pred[2]), int(pred[3])),
            color=(0, 128, 256),
            thickness=2,
        )
        cv2.putText(
            draw_img,
            "%s %.3f" % (lable_map[int(pred[4])], pred[5]),
            (int(pred[0]), int(pred[1]) - 10),
            color=(0, 128, 256),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
        )
  
    cv2.imwrite(save_path, draw_img) 

def save_result_with_gt(img, preds, gts, lable_map, save_path):
    img = (img-img.min())/(img.max()-img.min())
    img = (img * 255).astype("uint8")
    draw_img1 = np.zeros([img.shape[0], img.shape[1], 3])
    draw_img1[:, :, 0] = img
    draw_img1[:, :, 1] = img
    draw_img1[:, :, 2] = img
    draw_img2 = draw_img1.copy()

    for pred in preds:
        cv2.rectangle(
            draw_img1,
            (int(pred[0]), int(pred[1])),
            (int(pred[2]), int(pred[3])),
            color=(0, 128, 256),
            thickness=2,
        )
        cv2.putText(
            draw_img1,
            "%s %.3f" % (lable_map[int(pred[4])], pred[5]),
            (int(pred[0]), int(pred[1]) - 10),
            color=(0, 128, 256),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
        )

    for gt in gts:
        cv2.rectangle(
            draw_img2,
            (int(gt[0]), int(gt[1])),
            (int(gt[2]), int(gt[3])),
            color=(0, 200, 0),
            thickness=2,
        )
        cv2.putText(
            draw_img2,
            "%s" % (lable_map[gt[4]]),
            (int(gt[0]), int(gt[1]) - 10),
            color=(0, 128, 256),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.5,
        )

    draw_img = np.concatenate((draw_img1,draw_img2), 1)
    cv2.imwrite(save_path, draw_img)


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    model_detection = PredictModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_detection = Predictor(
        device=device,
        model=model_detection,
    )
    os.makedirs(output_path, exist_ok=True)
    label_map = {0:"knee"}
    img_dir = os.path.join(input_path, "imgs")
    label_dir = os.path.join(input_path, "labels")

    for sample in tqdm(os.listdir(img_dir)):  
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_dir, sample)))[0]
        try:
            gts = parse_label(label_map, os.path.join(label_dir, sample.replace(".dcm",".json")))
        except:
            gts = None
        preds = predictor_detection.predict(img)
        if gts is not None:        
            save_result_with_gt(img, preds, gts, label_map, os.path.join(output_path, f'{sample.replace(".dcm","")}.png'))
            # save_result(img, preds, lable_map, os.path.join(output_path, f'{sample.replace(".dcm","")}.png'))
            raw_result_dir = os.path.join(output_path, "raw_result")
            os.makedirs(raw_result_dir, exist_ok=True)
            # pred: [n, 6]  label:[n, 5]
            np.savez_compressed(os.path.join(raw_result_dir, sample.replace(".dcm",".npz")), pred=preds, label=gts)
        else:
            save_result(img, preds, label_map, os.path.join(output_path, f'{sample.replace(".dcm","")}.png'))



if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )