import numpy as np
import torch
import torch.nn as nn


# def generate_anchors(base_size=16, ratios=None, scales=None):
#     """
#     Generate anchor (reference) windows by enumerating aspect ratios X
#     scales w.r.t. a reference window.
#     """

#     if ratios is None:
#         ratios = np.array([0.5, 1, 2])

#     if scales is None:
#         scales = np.array([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

#     num_anchors = len(ratios) * len(scales)

#     # initialize output anchors
#     anchors = np.zeros((num_anchors, 4))

#     # scale base_size
#     anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

#     # compute areas of anchors
#     areas = anchors[:, 2] * anchors[:, 3]

#     # correct for ratios
#     anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
#     anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

#     # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
#     anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
#     anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

#     return anchors


# def shift(shape, stride, anchors):
#     shift_x = (np.arange(0, shape[1]) + 0.5) * stride
#     shift_y = (np.arange(0, shape[0]) + 0.5) * stride

#     shift_x, shift_y = np.meshgrid(shift_x, shift_y)


#     shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    

#     # add A anchors (1, A, 4) to
#     # cell K shifts (K, 1, 4) to get
#     # shift anchors (K, A, 4)
#     # reshape to (K*A, 4) shifted anchors
#     A = anchors.shape[0]
#     K = shifts.shape[0]

#     all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
#     all_anchors = all_anchors.reshape((K * A, 4))

#     return all_anchors


# anchors = generate_anchors()
# shifted_anchors = shift((128,128), 4, anchors)
# all_anchors = np.zeros((0, 4)).astype(np.float32)
# all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
# all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
# all_anchors = np.expand_dims(all_anchors, axis=0)
# print(all_anchors.shape)

# import os
# import sys
# import numpy as np
# from tqdm import tqdm
# from scipy.ndimage import zoom
# import SimpleITK as sitk
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pylab as plt
# import torch
# import cv2
# import random


# class random_rotate90(object):
#     def __init__(self, prob=0.5):
#         self.prob = prob

#     def __call__(self, img, label):
#         img_o, label_o = img, label
#         if random.random() < self.prob:
#             direction = random.choice([1, -1])
#             img_o = torch.rot90(img, direction, [1, 2])
#             H, W = img.size(1), img.size(2)
#             if direction == 1:
#                 # 顺时针旋转90度时，需要调整标注框的坐标
#                 x1 = label[:, 1]
#                 y1 = W - label[:, 2]
#                 x2 = label[:, 3]
#                 y2 = W - label[:, 0]
#                 label_o = torch.stack((x1, y1, x2, y2, label[:,4]), dim=1)
#             if direction == -1:  
#                 # 逆时针旋转90度时，需要调整标注框的坐标
#                 x1 = H - label[:, 3]
#                 y1 = label[:, 0]
#                 x2 = H - label[:, 1]
#                 y2 = label[:, 2]
#                 label_o = torch.stack((x1, y1, x2, y2, label[:,4]), dim=1)
#         return img_o, label_o


# class random_flip(object):
#     def __init__(self, axis=2, prob=0.5):
#         assert isinstance(axis, int) and axis in [1, 2]
#         self.axis = axis
#         self.prob = prob

#     def __call__(self, img, label):
#         img_o, label_o = img, label
#         if random.random() < self.prob:
#             _, height, width = img.shape
#             img_o = torch.flip(img, [self.axis])
#             if self.axis == 1:  # 对应于垂直翻转
#                 y1 = label[:, 1].clone()
#                 y2 = label[:, 3].clone()
                    
#                 label_o[:, 1] = height - 1 - y2
#                 label_o[:, 3] = height - 1 - y1

#             if self.axis == 2:  # 对应于水平翻转
#                 x1 = label[:, 0].clone()
#                 x2 = label[:, 2].clone()

#                 label_o[:, 0] = width - 1 - x2
#                 label_o[:, 2] = width - 1 - x1
                
#         return img_o, label_o

    

# file_name = r"E:\ShenFile\Code\03 Personal_Projects\目标检测标准框架\train\train_data\processed_data\1.npz"
# data = np.load(file_name, allow_pickle=True)
# img = data['img']
# label = data['label']

# img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# x1,y1,x2,y2,_ = label[0]
# print(x1,y1,x2,y2)
# cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
# # 显示图像
# cv2.imshow('Image with Boxes', img_show)

# img_o, label_o = random_flip(prob=1)(torch.from_numpy(img)[None], torch.from_numpy(label))
# label_o = label_o.numpy()
# img_o = img_o.numpy()[0]
# img_o = cv2.cvtColor(img_o, cv2.COLOR_GRAY2BGR)
# x1,y1,x2,y2,_ = label_o[0]
# print(x1,y1,x2,y2)
# cv2.rectangle(img_o, (x1, y1), (x2, y2), (0, 0, 255), 2)
# # 显示图像
# cv2.imshow('rotate', img_o)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


sorted_pred_bboxes = torch.Tensor([[ 35.,  18., 239., 220.],
              [ 34.,  19., 238., 223.],
              [ 33.,  19., 235., 220.], 
              [ 32.,  20., 234., 222.],
              [ 39.,  17., 244., 222.],
              [ 39.,  18., 244., 224.],
              [ 33.,  20., 238., 226.],
              [ 37.,  17., 240., 218.],
              [ 30.,  22., 234., 225.],
              [ 39.,  21., 243., 228.],
              [ 40.,  16., 245., 219.],
              [ 35.,  18., 237., 218.],
              [ 33.,  24., 238., 231.],
              [ 45.,  19., 249., 227.],
              [ 45.,  18., 249., 223.],
              [ 39.,  25., 244., 234.]]).cuda()

sorted_scores = torch.Tensor([0.9873, 0.9868, 0.9868, 0.9863, 0.9863, 0.9858, 0.9854, 0.9849, 0.9839,
        0.9839, 0.9829, 0.9829, 0.9824, 0.9819, 0.9814, 0.9814]).cuda()

nms_threshold = 0.1

keep = nms(sorted_pred_bboxes, sorted_scores, nms_threshold)

print(keep)