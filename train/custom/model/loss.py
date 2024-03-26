import torch
import torch.nn as nn

class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_thresh=0.5, neg_thresh=0.4, scale_factor=[0.1, 0.1, 0.2, 0.2]):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.scale_factor = scale_factor
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, classifications, regressions, anchors, labels):
        # print(torch.min(classifications),torch.max(classifications),torch.min(regressions),torch.max(regressions))
        device = classifications.device
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = labels[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones(classification.shape, device=device) * self.alpha

                alpha_factor = 1.0 - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0, device=device).float())
                continue

            IoU = self.calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_labels

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            targets = torch.ones(classification.shape, device=device) * -1
            negative_indices = torch.lt(IoU_max, self.neg_thresh)
            targets[negative_indices, :] = 0

            positive_indices = torch.ge(IoU_max, self.pos_thresh)
            num_positive_anchors = positive_indices.sum()

            assigned_labels = bbox_annotation[IoU_argmax, :]
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_labels[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape, device=device) * self.alpha
            alpha_factor = torch.where(torch.eq(targets, 1.0), alpha_factor, 1.0 - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.0), 1.0 - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            # bce = self.bce_loss(classification, targets)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=device))
            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_labels = assigned_labels[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_labels[:, 2] - assigned_labels[:, 0]
                gt_heights = assigned_labels[:, 3] - assigned_labels[:, 1]
                gt_ctr_x = assigned_labels[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_labels[:, 1] + 0.5 * gt_heights

                # 限制框的长宽至少为1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                # 计算每个anchor对应的预测目标：平移和缩放参数
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                # 将预测目标进行标准化，使用了预设的缩放因子
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()
                if self.scale_factor:
                    targets = targets / torch.tensor([self.scale_factor], device=device)

                # smooth L1损失函数
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0,
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0, device=device).float())

        return {"c_loss":torch.stack(classification_losses).mean(), "r_loss":torch.stack(regression_losses).mean()} 

    def calc_iou(self, a, b):
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

        ua = torch.clamp(ua, min=1e-8)

        intersection = iw * ih

        IoU = intersection / ua

        return IoU