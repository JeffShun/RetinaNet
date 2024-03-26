import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

class Decoder(nn.Module):
    def __init__(
        self, image_w, image_h, scale_factor, top_n=1000, min_score_threshold=0.05, nms_threshold=0.5, max_detection_num=100
    ):
        super(Decoder, self).__init__()
        self.image_w = image_w  # 图像的宽度
        self.image_h = image_h  # 图像的高度
        self.scale_factor = scale_factor  # 缩放因子
        self.top_n = top_n  # 选择分数最高的前n个候选框
        self.min_score_threshold = min_score_threshold  # 分数阈值，低于此阈值的框将被丢弃
        self.nms_threshold = nms_threshold  # 非极大值抑制的IoU阈值
        self.max_detection_num = max_detection_num  # 每张图像最大检测框数量

    def forward(self, cls_heads, reg_heads, batch_anchors):
        # nbox=(w//8*h//8)+(w//16*h//16)+(w//32+h//32)*n_anchor
        # cls_heads: [batch, nbox, classes]
        # reg_heads: [batch, nbox, 4]
        # batch_anchors: [batch, nbox, 4]

        device = cls_heads[0].device 
        with torch.no_grad():
            # 获取分类头和回归头的预测结果
            filter_scores, filter_score_classes = torch.max(cls_heads, dim=2)  # 选择分数最高的类别
            filter_scores, indexes = torch.topk(filter_scores, self.top_n, dim=1, largest=True, sorted=True)  # 选择top_n个预测结果
            filter_score_classes = torch.gather(filter_score_classes, 1, indexes)                             # shape: [batch, top_n]
            filter_reg_heads = torch.gather(reg_heads, 1, indexes.unsqueeze(-1).repeat(1, 1, 4))              # shape: [batch, top_n, 4]
            filter_batch_anchors = torch.gather(batch_anchors, 1, indexes.unsqueeze(-1).repeat(1, 1, 4))      # shape: [batch, top_n, 4]

            # 存储每张图片的结果
            batch_scores, batch_classes, batch_pred_bboxes = [], [], []
            for per_image_scores, per_image_score_classes, per_image_reg_heads, per_image_anchors in zip(
                filter_scores, filter_score_classes, filter_reg_heads, filter_batch_anchors
            ):
                # 将回归头的预测值转换为边界框坐标
                pred_bboxes = self.snap_tx_ty_tw_th_reg_heads_to_x1_y1_x2_y2_bboxes(per_image_reg_heads, per_image_anchors)
                # 根据预测分数阈值筛选预测结果
                score_classes = per_image_score_classes[per_image_scores > self.min_score_threshold].float()
                pred_bboxes = pred_bboxes[per_image_scores > self.min_score_threshold].float()
                scores = per_image_scores[per_image_scores > self.min_score_threshold].float()

                # 初始化每张图片的结果
                one_image_scores = (-1) * torch.ones((self.max_detection_num,), device=device)
                one_image_classes = (-1) * torch.ones((self.max_detection_num,), device=device)
                one_image_pred_bboxes = (-1) * torch.ones((self.max_detection_num, 4), device=device)

                if scores.shape[0] != 0:
                    # 非极大值抑制，去除重叠较多的检测框
                    sorted_scores, sorted_indexes = torch.sort(scores, descending=True)
                    sorted_score_classes = score_classes[sorted_indexes]
                    sorted_pred_bboxes = pred_bboxes[sorted_indexes]
                    keep = nms(sorted_pred_bboxes, sorted_scores, self.nms_threshold)
                    keep_scores = sorted_scores[keep]
                    keep_classes = sorted_score_classes[keep]
                    keep_pred_bboxes = sorted_pred_bboxes[keep]

                    # 限制最大检测框数量
                    final_detection_num = min(self.max_detection_num, keep_scores.shape[0])
                    one_image_scores[0:final_detection_num] = keep_scores[0:final_detection_num]
                    one_image_classes[0:final_detection_num] = keep_classes[0:final_detection_num]
                    one_image_pred_bboxes[0:final_detection_num, :] = keep_pred_bboxes[0:final_detection_num, :]

                one_image_scores = one_image_scores.unsqueeze(0)
                one_image_classes = one_image_classes.unsqueeze(0)
                one_image_pred_bboxes = one_image_pred_bboxes.unsqueeze(0)

                # 将每张图片的结果存储到批次结果中
                batch_scores.append(one_image_scores)
                batch_classes.append(one_image_classes)
                batch_pred_bboxes.append(one_image_pred_bboxes)

            # 拼接批次结果
            batch_scores = torch.cat(batch_scores, dim=0)
            batch_classes = torch.cat(batch_classes, dim=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, dim=0)
            
            return batch_scores, batch_classes, batch_pred_bboxes

    def snap_tx_ty_tw_th_reg_heads_to_x1_y1_x2_y2_bboxes(self, reg_heads, anchors):
        """
        将回归头的预测值转换为边界框坐标
        reg_heads:[anchor_nums,4],4:[tx,ty,tw,th]
        anchors:[anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        """
        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

        if self.scale_factor:
            factor = torch.tensor([self.scale_factor]).to(anchors.device)
            reg_heads = reg_heads * factor

        pred_bboxes_wh = torch.exp(reg_heads[:, 2:]) * anchors_wh
        pred_bboxes_ctr = reg_heads[:, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_min_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_max_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = torch.cat([pred_bboxes_x_min_y_min, pred_bboxes_x_max_y_max], dim=1)
        pred_bboxes = pred_bboxes.int()

        # 将边界框坐标限制在图像范围内
        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2], max=self.image_w - 1)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3], max=self.image_h - 1)

        return pred_bboxes
