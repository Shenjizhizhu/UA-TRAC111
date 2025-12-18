import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi


class YOLOv5Loss(nn.Module):
    def __init__(self, num_classes, img_size=640, anchors=None):
        super(YOLOv5Loss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.eps = 1e-12
        if anchors is None:
            self.anchors = torch.tensor([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])
        else:
            self.anchors = anchors
        self.anchors = self.anchors.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_anchors = self.anchors.shape[0]

        self.bbox_weight = 1.0
        self.conf_weight = 0.5
        self.cls_weight = 0.2
        self.pos_weight = 1.0
        self.neg_weight = 0.1
        self.small_target_weight = 1.0

    def ciou_boxes(self, box1, box2):
        inter_x1 = torch.max(box1[0], box2[0])
        inter_y1 = torch.max(box1[1], box2[1])
        inter_x2 = torch.min(box1[2], box2[2])
        inter_y2 = torch.min(box1[3], box2[3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area + self.eps
        iou = inter_area / union_area

        center_dist = torch.square(box1[0] + box1[2] - box2[0] - box2[2]) / 4 + torch.square(box1[1] + box1[3] - box2[1] - box2[3]) / 4
        enclose_x1 = torch.min(box1[0], box2[0])
        enclose_y1 = torch.min(box1[1], box2[1])
        enclose_x2 = torch.max(box1[2], box2[2])
        enclose_y2 = torch.max(box1[3], box2[3])
        enclose_dist = torch.square(enclose_x2 - enclose_x1) + torch.square(enclose_y2 - enclose_y1) + self.eps
        cd = center_dist / enclose_dist

        v = (4 / (pi ** 2)) * torch.square(torch.atan((box1[2] - box1[0]) / (box1[3] - box1[1] + self.eps)) - torch.atan((box2[2] - box2[0]) / (box2[3] - box2[1] + self.eps)))
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        ciou = iou - cd - alpha * v
        return ciou

    def match_anchors(self, target_boxes):
        batch_size = target_boxes.shape[0]
        max_boxes = target_boxes.shape[1]
        match_indices = torch.zeros((batch_size, max_boxes), dtype=torch.long, device=target_boxes.device)
        
        valid_mask = (target_boxes[..., 2] > 1e-6) & (target_boxes[..., 3] > 1e-6)
        valid_boxes = target_boxes[valid_mask]
        if valid_boxes.numel() == 0:
            return match_indices

        box_xyxy = torch.stack([
            valid_boxes[..., 0] - valid_boxes[..., 2]/2,
            valid_boxes[..., 1] - valid_boxes[..., 3]/2,
            valid_boxes[..., 0] + valid_boxes[..., 2]/2,
            valid_boxes[..., 1] + valid_boxes[..., 3]/2
        ], dim=-1) * self.img_size

        anchor_xyxy = torch.cat([
            torch.zeros((self.num_anchors, 2), device=box_xyxy.device),
            self.anchors.to(box_xyxy.device)
        ], dim=-1)

        ciou_scores = []
        for box in box_xyxy:
            ciou = self.ciou_boxes(box.unsqueeze(0).repeat(self.num_anchors, 1), anchor_xyxy)
            ciou_scores.append(ciou)
        ciou_scores = torch.stack(ciou_scores, dim=0)
        match_idx = torch.argmax(ciou_scores, dim=1)
        match_indices[valid_mask] = match_idx
        return match_indices

    def mse_bbox_loss(self, pred_boxes, target_boxes):
        valid_mask = (target_boxes[..., 2] > 1e-6) & (target_boxes[..., 3] > 1e-6)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_boxes.device)

        pred_boxes_valid = pred_boxes[valid_mask]
        target_boxes_valid = target_boxes[valid_mask]

        scale_factor = self.img_size / 1000
        pred_boxes_pixel = pred_boxes_valid * scale_factor
        target_boxes_pixel = target_boxes_valid * scale_factor

        mse_loss = F.mse_loss(pred_boxes_pixel, target_boxes_pixel, reduction='mean')

        target_wh = target_boxes_valid[..., 2:] * self.img_size
        target_area = target_wh[..., 0] * target_wh[..., 1]
        small_target_mask = target_area < 32*32
        if small_target_mask.any():
            small_pred = pred_boxes_pixel[small_target_mask]
            small_target = target_boxes_pixel[small_target_mask]
            small_mse = F.mse_loss(small_pred, small_target, reduction='mean')
            mse_loss = 0.999 * mse_loss + 0.001 * small_mse * self.small_target_weight

        return mse_loss

    def forward(self, preds, target_boxes, target_labels):
        self.match_anchors(target_boxes)
        
        min_box_num = min(preds.shape[1], target_boxes.shape[1])
        pred_boxes = preds[:, :min_box_num, :4]
        pred_conf = preds[:, :min_box_num, 4:5]
        pred_cls = preds[:, :min_box_num, 5:]
        target_boxes = target_boxes[:, :min_box_num, :]
        target_labels = target_labels[:, :min_box_num]

        target_boxes = torch.nan_to_num(target_boxes, nan=0.0, posinf=1.0, neginf=0.0)
        pred_boxes = torch.nan_to_num(pred_boxes, nan=0.0, posinf=1.0, neginf=0.0)

        valid_mask = (target_boxes[..., 2] > 1e-6) & (target_boxes[..., 3] > 1e-6)
        bbox_loss = self.mse_bbox_loss(pred_boxes, target_boxes) if valid_mask.any() else torch.tensor(0.0, device=preds.device)

        target_conf = valid_mask.float().unsqueeze(-1)
        weight = torch.where(valid_mask.unsqueeze(-1), self.pos_weight, self.neg_weight)
        conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, weight=weight, reduction='mean'
        )

        if self.num_classes == 1:
            pred_cls = pred_cls.squeeze(-1)
            target_cls = target_labels.float()
            cls_weight = valid_mask.float() * self.pos_weight + (1 - valid_mask.float()) * self.neg_weight
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_cls, target_cls, weight=cls_weight, reduction='mean'
            )
        else:
            cls_loss = F.cross_entropy(
                pred_cls.reshape(-1, self.num_classes), 
                target_labels.reshape(-1), 
                reduction='none'
            )
            cls_loss = cls_loss.reshape(target_labels.shape)
            cls_loss = torch.where(valid_mask, cls_loss, torch.zeros_like(cls_loss)).mean()

        total_loss = self.bbox_weight * bbox_loss + self.conf_weight * conf_loss + self.cls_weight * cls_loss
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1.0, neginf=0.0)
        
        return total_loss, bbox_loss, conf_loss, cls_loss