"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional
import os
import numpy as np
from runstats import Statistics
import pandas as pd

def iou(box1, box2):
    x1,y1, x2, y2 = box1[:4]
    a1,b1, a2, b2 = box2[:4]

    ax = max(x1, a1)
    ay = max(y1, b1) 
    bx = min(x2, a2) 
    by = min(y2, b2) 
	
    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)
	
    w = bx - ax
    h = by - ay
    if w<=0 or h<=0:
        return 0 
    area_X = w * h
    return area_X / (area_N + area_M - area_X)

def metric_func(pred, gt, valid_label, iou_threshold):
    TP = 0
    FP = 0
    AP = 0
    for box1 in gt:
        if int(box1[4]) == valid_label:
            AP += 1
            for box2 in pred:
                if int(box2[4]) == valid_label:
                    if iou(box1, box2) > iou_threshold:
                        TP += 1
                        break
                    else:
                        FP += 1
                        continue
    return TP, FP, AP

def recall(pred, gt, valid_label, iou_threshold):
    TP, FP, AP = metric_func(pred, gt, valid_label, iou_threshold)
    return TP / AP


def precision(pred, gt, valid_label, iou_threshold):
    TP, FP, AP = metric_func(pred, gt, valid_label, iou_threshold)
    return TP / (TP + FP)


METRIC_FUNCS = dict(
    RECALL=recall,
    PRECISION=precision,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}
        self.metrics_data = {metric:[] for metric in metric_funcs}

    def push(self, pid, pred, label, valid_label, iou_threshold):
        for metric, func in METRIC_FUNCS.items():
            val = func(pred, label, valid_label, iou_threshold)
            self.metrics[metric].push(val)
            self.metrics_data[metric].append((pid, val))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}
    

    def save(self, save_dir):
        os.makedirs(save_dir,exist_ok=True)
        df = pd.DataFrame()
        # 遍历数据字典，将每种评价方式的数据添加为DataFrame的一列
        for method, values in self.metrics_data.items():
            labels, scores = zip(*values)
            df[method] = scores
        df['pid'] = labels
        df = df[['pid'] + [col for col in df.columns if col != 'pid']]
        csv_file_path = save_dir + "/metrics.csv"
        df.to_csv(csv_file_path, index=False)


    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}"
            for name in metric_names
        )

def evaluate(args):
    metrics = Metrics(METRIC_FUNCS)
    valid_label = args.valid_label
    iou_threshold = args.iou_threshold
    for sample in args.data_path.iterdir():
        pid = str(sample).split("\\")[-1].replace("npz","")
        data = np.load(sample, allow_pickle=True)
        pred = data['pred']
        label = data['label']
        metrics.push(pid, pred, label, valid_label, iou_threshold)

    return metrics


if __name__ == "__main__":
    
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data_path", type=pathlib.Path,
        default=pathlib.Path("./data/output/raw_result"),
    )
    parser.add_argument(
        "--valid_label", type=int,
        default=0,
    )
    parser.add_argument(
        "--iou_threshold", type=float,
        default=0.2,
    )
    args = parser.parse_args()
    metrics = evaluate(args)
    metrics.save("./data/output/metric_data")
    print(metrics)
