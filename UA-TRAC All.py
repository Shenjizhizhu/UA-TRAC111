import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet34_Weights
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from math import pi
from collections import defaultdict
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def draw_original_car_labels():
    path = os.path.join('/', 'home', 'adduser', 'Shenji', 'UA-TRAC')
    save_dir = os.path.join(path, 'outputting_imgs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gt_path = os.path.join(path, 'train_gt.txt')
    with open(gt_path, 'r') as f:
        gt = f.readlines()

    num = 20
    pick_idx = range(min(num, len(gt)))

    for idx in pick_idx:
        line = gt[idx].strip().split()
        if not line:
            continue
        name = line[0]
        file = os.path.join(path, 'Insight-MVT_Annotation_Train', name)

        img = cv2.imread(file)
        if img is None:
            print(f"警告：无法读取图片 {file}，跳过")
            continue

        if len(line) > 1:
            boxes = np.array([float(x) for x in line[1:]]).reshape(-1, 4)
            for x1, y1, x2, y2 in boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
        save_name = name.replace(os.sep, '_') + '_original.jpg'
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, img)
    print(f"原始标签框绘制完成，保存至：{save_dir}")

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

def load_gt_data(gt_path):
    with open(gt_path, 'r') as f:
        gt_data = [line.strip() for line in f if line.strip()]
    return gt_data

def shuffle_and_split_gt(gt_data, test_ratio=0.2, val_ratio=0.1, save_root='./'):
    train_val_data, test_data = train_test_split(
        gt_data, test_size=test_ratio, random_state=SEED, shuffle=True
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_ratio/(1-test_ratio), random_state=SEED, shuffle=True
    )
    
    for name, data in zip(['train_gt_new', 'val_gt_new', 'test_gt_new'], [train_data, val_data, test_data]):
        save_path = os.path.join(save_root, f'{name}.txt')
        with open(save_path, 'w') as f:
            f.write('\n'.join(data))
    print(f"数据划分完成！训练集：{len(train_data)} 条，验证集：{len(val_data)} 条，测试集：{len(test_data)} 条")
    return (
        os.path.join(save_root, 'train_gt_new.txt'),
        os.path.join(save_root, 'val_gt_new.txt'),
        os.path.join(save_root, 'test_gt_new.txt')
    )

def load_boxes_from_gt(gt_path, img_root, img_size=640):
    gt_data = load_gt_data(gt_path)
    boxes_wh = []
    for line in gt_data:
        parts = line.split()
        if len(parts) < 2:
            continue
        img_path = os.path.join(img_root, parts[0])
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_ori, w_ori = img.shape[:2]
        boxes = np.array([float(x) for x in parts[1:]]).reshape(-1, 4)
        valid_mask = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3]) & \
                     (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes = boxes[valid_mask]
        boxes = np.clip(boxes, 0, max(w_ori, h_ori))
        w = (boxes[:, 2] - boxes[:, 0]) * (img_size / w_ori)
        h = (boxes[:, 3] - boxes[:, 1]) * (img_size / h_ori)
        boxes_wh.extend(zip(w, h))
    return np.array(boxes_wh)

def kmeans_anchors(boxes_wh, n_clusters=9, img_size=640):
    if len(boxes_wh) == 0:
        return np.array([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=20)
    kmeans.fit(boxes_wh)
    anchors = kmeans.cluster_centers_
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    anchors = np.clip(anchors, 1, img_size)
    return anchors.astype(np.int32)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compute_ap(precision, recall):
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap

def evaluate_mAP(pred_boxes_list, gt_boxes_list, iou_thres=0.5):
    all_preds = defaultdict(list)
    all_gts = defaultdict(list)
    gt_counts = defaultdict(int)

    for img_idx, (pred_boxes, gt_boxes) in enumerate(zip(pred_boxes_list, gt_boxes_list)):
        for pb in pred_boxes:
            all_preds[0].append((img_idx, pb[4], pb[:4]))
        for gb in gt_boxes:
            all_gts[0].append((img_idx, gb))
            gt_counts[0] += 1

    ap_list = []
    for cls in all_preds.keys():
        if gt_counts[cls] == 0:
            ap_list.append(0.0)
            continue
        preds = sorted(all_preds[cls], key=lambda x: x[1], reverse=True)
        gts = all_gts[cls]
        gt_matched = set()
        precision = []
        recall = []
        tp = 0
        fp = 0

        for img_idx, conf, box in preds:
            matched = False
            for gt_idx, (gt_img_idx, gt_box) in enumerate(gts):
                if gt_img_idx == img_idx and (img_idx, gt_idx) not in gt_matched:
                    iou = calculate_iou(box, gt_box)
                    if iou >= iou_thres:
                        tp += 1
                        gt_matched.add((img_idx, gt_idx))
                        matched = True
                        break
            if not matched:
                fp += 1
            precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recall.append(tp / gt_counts[cls] if gt_counts[cls] > 0 else 0.0)

        ap = compute_ap(np.array(precision), np.array(recall))
        ap_list.append(ap)

    mAP = np.mean(ap_list) if ap_list else 0.0
    return mAP

def custom_collate_fn(batch):
    images = []
    target_boxes_list = []
    target_labels_list = []
    img_paths = []
    original_boxes_list = []

    for item in batch:
        img, (t_boxes, t_labels), img_path, orig_boxes = item
        images.append(img)
        target_boxes_list.append(t_boxes)
        target_labels_list.append(t_labels)
        img_paths.append(img_path)
        original_boxes_list.append(orig_boxes)

    images = torch.stack(images, 0)
    max_boxes = max([b.shape[0] for b in target_boxes_list])
    padded_boxes = []
    padded_labels = []
    for b, l in zip(target_boxes_list, target_labels_list):
        if b.shape[0] < max_boxes:
            pad_box = torch.zeros((max_boxes - b.shape[0], 4), dtype=b.dtype, device=b.device)
            pad_label = torch.zeros(max_boxes - l.shape[0], dtype=l.dtype, device=l.device)
            b = torch.cat([b, pad_box], dim=0)
            l = torch.cat([l, pad_label], dim=0)
        padded_boxes.append(b)
        padded_labels.append(l)
    target_boxes = torch.stack(padded_boxes, 0)
    target_labels = torch.stack(padded_labels, 0)

    return images, (target_boxes, target_labels), img_paths, original_boxes_list

class UATracDatasetYOLO(Dataset):
    def __init__(self, gt_path, img_root, transform=None, num_classes=1, max_boxes=20, img_size=640, max_samples=None):
        self.gt_data = load_gt_data(gt_path)
        if max_samples is not None and max_samples > 0:
            self.gt_data = self.gt_data[:max_samples]
        self.img_root = img_root
        self.transform = transform
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.img_size = img_size 
        if self.num_classes < 1:
            raise ValueError(f"类别数num_classes必须≥1，当前值：{self.num_classes}")

    def __len__(self):
        return len(self.gt_data)

    def __getitem__(self, idx):
        line = self.gt_data[idx].split()
        img_path = os.path.join(self.img_root, line[0])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"图片文件不存在或无法读取：{img_path}")
        img_ori = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_ori, w_ori = img.shape[:2]

        if len(line) < 2:
            boxes = np.zeros((0, 4))
        else:
            boxes = np.array([float(x) for x in line[1:]]).reshape(-1, 4)
            valid_mask = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3]) & \
                         (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
            boxes = boxes[valid_mask]
            boxes = np.clip(boxes, 0, max(w_ori, h_ori))

        labels = np.zeros(len(boxes), dtype=np.int64)
        original_boxes = boxes.copy()
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        h_trans, w_trans = self.img_size, self.img_size

        yolo_boxes = np.zeros((min(len(boxes), self.max_boxes), 4), dtype=np.float32)
        yolo_labels = np.zeros(min(len(boxes), self.max_boxes), dtype=np.int64)
        
        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx = ((x1 + x2) / 2) / w_trans
            cy = ((y1 + y2) / 2) / h_trans
            w = (x2 - x1) / w_trans
            h = (y2 - y1) / h_trans
            cx = np.clip(cx, 1e-4, 1 - 1e-4)
            cy = np.clip(cy, 1e-4, 1 - 1e-4)
            w = np.clip(w, 1e-4, 1 - 1e-4)
            h = np.clip(h, 1e-4, 1 - 1e-4)
            take = min(len(boxes), self.max_boxes)
            yolo_boxes[:take] = np.stack([cx, cy, w, h], axis=1)[:take]
            yolo_labels[:take] = labels[:take]

        return (
            img,
            (torch.from_numpy(yolo_boxes).float(), torch.from_numpy(yolo_labels).long()),
            img_path,
            original_boxes
        )

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01)
        ) for in_ch in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.01)
        ) for _ in in_channels_list])
        self.small_target_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        for m in self.lateral_convs + self.fpn_convs + [self.small_target_branch]:
            for sub_m in m:
                if isinstance(sub_m, nn.Conv2d):
                    nn.init.kaiming_normal_(sub_m.weight, mode='fan_out', nonlinearity='relu')
                    if sub_m.bias is not None:
                        nn.init.constant_(sub_m.bias, 0)
                elif isinstance(sub_m, nn.BatchNorm2d):
                    nn.init.constant_(sub_m.weight, 1)
                    nn.init.constant_(sub_m.bias, 0)

    def forward(self, x_list):
        lateral_feats = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, x_list)]
        fpn_feats = []
        prev_feat = lateral_feats[-1]
        fpn_feats.append(self.fpn_convs[-1](prev_feat))
        for i in range(len(lateral_feats)-2, -1, -1):
            prev_feat = self.upsample(prev_feat)
            if prev_feat.shape[2:] != lateral_feats[i].shape[2:]:
                prev_feat = F.interpolate(prev_feat, size=lateral_feats[i].shape[2:], mode='bilinear', align_corners=True)
            prev_feat = prev_feat + lateral_feats[i]
            fpn_feats.insert(0, self.fpn_convs[i](prev_feat))
        
        fpn_feats[0] = self.small_target_branch(fpn_feats[0])
        return fpn_feats

class YOLOv5Detect(nn.Module):
    def __init__(self, num_classes, in_channels, img_size=640, conf_filter_thres=0.25, anchors=None):
        super(YOLOv5Detect, self).__init__()
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes
        self.img_size = img_size
        self.conf_filter_thres = conf_filter_thres
        if anchors is None:
            self.anchors = torch.tensor([[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]])
        else:
            self.anchors = anchors
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.001),
            nn.Conv2d(in_ch * 2, in_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, self.num_outputs, kernel_size=1)
        ) for in_ch in in_channels])

        for conv_seq in self.convs:
            for conv in conv_seq:
                if isinstance(conv, nn.Conv2d):
                    if conv.out_channels == self.num_outputs:
                        b = conv.bias.view(-1, self.num_outputs)
                        b.data[:, 4] = torch.log(torch.tensor(1.0 / (1 - self.conf_filter_thres)))
                        b.data[:, 5:] = torch.log(torch.tensor(1.0 / self.num_classes))
                        conv.bias = torch.nn.Parameter(b.view(-1))
                    else:
                        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                        if conv.bias is not None:
                            nn.init.constant_(conv.bias, 0)
                elif isinstance(conv, nn.BatchNorm2d):
                    nn.init.constant_(conv.weight, 1)
                    nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        preds = []
        for feat, conv_seq in zip(x, self.convs):
            batch_size, _, h, w = feat.shape
            pred = conv_seq(feat).permute(0, 2, 3, 1).reshape(batch_size, h*w, self.num_outputs)
            if self.training:
                preds.append(pred)
            else:
                pred = torch.sigmoid(pred)
                batch_preds = []
                for b in range(batch_size):
                    single_pred = pred[b]
                    mask = single_pred[..., 4] >= self.conf_filter_thres
                    batch_preds.append(single_pred[mask])
                max_len = max([len(p) for p in batch_preds]) if batch_preds else 0
                padded_preds = []
                for p in batch_preds:
                    if len(p) < max_len:
                        pad = torch.zeros((max_len - len(p), self.num_outputs), device=p.device)
                        p = torch.cat([p, pad], dim=0)
                    padded_preds.append(p.unsqueeze(0))
                pred = torch.cat(padded_preds, dim=0) if padded_preds else torch.zeros((batch_size, 0, self.num_outputs), device=feat.device)
                preds.append(pred)

        preds = torch.cat(preds, dim=1)
        if not self.training and preds.shape[1] > 1000:
            preds = preds[:, :1000, :]
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
        return preds

class ResNetFPNYOLOv5(nn.Module):
    def __init__(self, num_classes=1, img_size=640, anchors=None):
        super(ResNetFPNYOLOv5, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        for idx, param in enumerate(self.backbone.parameters()):
            if idx < 20:
                param.requires_grad = False
            else:
                param.requires_grad = True
        self.fpn_in_channels = [128, 256, 512]
        self.fpn_out_channels = 256 
        self.fpn = FPN(self.fpn_in_channels, self.fpn_out_channels)
        self.detect = YOLOv5Detect(
            num_classes=num_classes,
            in_channels=[self.fpn_out_channels]*3,
            img_size=img_size,
            conf_filter_thres=0.25,
            anchors=anchors
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        fpn_feats = self.fpn([c2, c3, c4])
        preds = self.detect(fpn_feats)
        return preds

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
        _ = self.match_anchors(target_boxes)
        
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
            pred_conf, 
            target_conf, 
            weight=weight,
            reduction='mean'
        )

        if self.num_classes == 1:
            pred_cls = pred_cls.squeeze(-1)
            target_cls = target_labels.float()
            cls_weight = valid_mask.float() * self.pos_weight + (1 - valid_mask.float()) * self.neg_weight
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_cls, 
                target_cls, 
                weight=cls_weight,
                reduction='mean'
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, root_path, epochs=20, patience=5, grad_clip=5.0, accumulate_steps=2):
    best_val_loss = float('inf')
    best_mAP = 0.0
    early_stop_count = 0

    for epoch in range(epochs):
        model.train()
        train_losses = [0.0, 0.0, 0.0, 0.0]
        optimizer.zero_grad()
        batch_num = len(train_loader)
        
        for batch_idx, (images, (target_boxes, target_labels), _, _) in enumerate(train_loader):
            images = images.to(device)
            target_boxes = target_boxes.to(device)
            target_labels = target_labels.to(device)

            preds = model(images)
            total_loss, bbox_loss, conf_loss, cls_loss = criterion(preds, target_boxes, target_labels)
            (total_loss / accumulate_steps).backward()

            if (batch_idx + 1) % accumulate_steps == 0 or (batch_idx + 1) == batch_num:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_losses[0] += total_loss.item()
            train_losses[1] += bbox_loss.item()
            train_losses[2] += conf_loss.item()
            train_losses[3] += cls_loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{batch_num}], '\
                      f'Total Loss: {total_loss.item():.8f}, Bbox Loss: {bbox_loss.item():.8f}, '\
                      f'Conf Loss: {conf_loss.item():.8f}, Cls Loss: {cls_loss.item():.8f}')

        model.eval()
        val_loss = 0.0
        pred_boxes_list = []
        gt_boxes_list = []
        
        with torch.no_grad():
            for images, (target_boxes, target_labels), _, original_boxes_list in val_loader:
                images = images.to(device)
                target_boxes = target_boxes.to(device)
                target_labels = target_labels.to(device)
                
                preds = model(images)
                total_loss, _, _, _ = criterion(preds, target_boxes, target_labels)
                
                val_loss += total_loss.item()

                preds_np = preds.cpu().numpy()
                for i in range(len(preds_np)):
                    pred = preds_np[i]
                    pred = pred[pred[:, 4] >= 0.3]
                    pred_boxes = []
                    for p in pred:
                        cx, cy, w, h, conf = p[:5]
                        x1 = (cx - w/2) * model.img_size
                        y1 = (cy - h/2) * model.img_size
                        x2 = (cx + w/2) * model.img_size
                        y2 = (cy + h/2) * model.img_size
                        pred_boxes.append([x1, y1, x2, y2, conf])
                    pred_boxes_list.append(pred_boxes)
                    gt_boxes_list.append(original_boxes_list[i].tolist())

        train_avg_loss = train_losses[0] / batch_num
        val_avg_loss = val_loss / len(val_loader)
        val_mAP = evaluate_mAP(pred_boxes_list, gt_boxes_list, iou_thres=0.5)

        if isinstance(scheduler, SequentialLR):
            scheduler.step()
        else:
            scheduler.step(val_avg_loss)

        print(f'\nEpoch [{epoch+1}/{epochs}] Summary:')
        print(f'Train Avg Loss: {train_avg_loss:.8f}, Val Avg Loss: {val_avg_loss:.8f}, Val mAP@0.5: {val_mAP:.4f}')
        print(f'Current LR: {optimizer.param_groups[0]["lr"]:.6f}\n')

        if val_mAP > best_mAP or (val_avg_loss < best_val_loss and val_mAP >= best_mAP - 0.01):
            best_val_loss = min(val_avg_loss, best_val_loss)
            best_mAP = max(val_mAP, best_mAP)
            early_stop_count = 0
            torch.save(model.state_dict(), os.path.join(root_path, 'best_yolov5_resnet_fpn.pth'))
            print(f'Best model saved (Val Loss: {best_val_loss:.8f}, Val mAP: {best_mAP:.4f})')
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break

    torch.save(model.state_dict(), os.path.join(root_path, 'last_yolov5_resnet_fpn.pth'))
    print('训练完成！')
    return model

def non_max_suppression(boxes, scores, conf_thres=0.5, iou_thres=0.25):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    high_conf_mask = scores >= conf_thres
    boxes = boxes[high_conf_mask]
    scores = scores[high_conf_mask]
    if len(boxes) == 0:
        return []

    boxes_xywh = np.stack([
        boxes[:, 0], boxes[:, 1],
        boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    ], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_thres, iou_thres)

    if isinstance(indices, int):
        indices = [indices] if indices >= 0 else []
    else:
        indices = indices.flatten().tolist() if len(indices) > 0 else []
    return indices

def test_model_with_draw(model_path, test_loader, device, img_size=640, conf_thres=0.5, iou_thres=0.25, anchors=None):
    model = ResNetFPNYOLOv5(num_classes=1, img_size=img_size, anchors=anchors).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"成功加载模型：{model_path}")

    save_dir = os.path.join('/', 'home', 'adduser', 'Shenji', 'UA-TRAC', 'outputting_imgs')
    create_dir(save_dir)

    class_names = ['car']
    max_test_batches = 10 
    pred_boxes_list = []
    gt_boxes_list = []

    for batch_idx, (images, (_, _), img_paths, original_boxes_list) in enumerate(test_loader):
        if batch_idx >= max_test_batches:
            break
        images = images.to(device)
        with torch.no_grad():
            preds = model(images)

        preds_np = preds.cpu().numpy()
        for i in range(len(preds_np)):
            pred = preds_np[i]
            img_path = img_paths[i]
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告：无法读取图片 {img_path}，跳过")
                pred_boxes_list.append([])
                gt_boxes_list.append(original_boxes_list[i].tolist())
                continue
            img_copy = img.copy()
            h_img, w_img = img_copy.shape[:2]

            original_boxes = original_boxes_list[i]
            gt_boxes = []
            pred_boxes = []
            if len(original_boxes) > 0:
                for x1, y1, x2, y2 in original_boxes:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1 = max(0, min(x1, w_img-1))
                    y1 = max(0, min(y1, h_img-1))
                    x2 = max(0, min(x2, w_img-1))
                    y2 = max(0, min(y2, h_img-1))
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_copy, 'GT', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    gt_boxes.append([x1, y1, x2, y2])

                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img_copy, f'{class_names[0]} 1.00', (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    pred_boxes.append([x1, y1, x2, y2, 1.0])
            gt_boxes_list.append(gt_boxes)
            pred_boxes_list.append(pred_boxes)

            if len(gt_boxes) == 0 and len(pred) > 0:
                pred = pred[pred[:, 4] >= conf_thres]
                if len(pred) > 0:
                    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
                    x1 = (cx - w/2) * img_size
                    y1 = (cy - h/2) * img_size
                    x2 = (cx + w/2) * img_size
                    y2 = (cy + h/2) * img_size
                    scale_x = w_img / img_size
                    scale_y = h_img / img_size
                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y
                    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
                    scores = pred[:, 4].tolist()
                    indices = non_max_suppression(boxes, scores, conf_thres, iou_thres)
                    for idx in indices:
                        idx = int(idx)
                        if idx >= len(boxes):
                            continue
                        x1, y1, x2, y2 = map(int, boxes[idx])
                        x1 = max(0, min(x1, w_img-1))
                        y1 = max(0, min(y1, h_img-1))
                        x2 = max(0, min(x2, w_img-1))
                        y2 = max(0, min(y2, h_img-1))
                        conf = scores[idx]
                        label = f'{class_names[0]} {conf:.2f}'
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        pred_boxes.append([x1, y1, x2, y2, conf])
                    pred_boxes_list[-1] = pred_boxes

            img_name = os.path.basename(img_path).replace('.jpg', '_pred.jpg')
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, img_copy)
            print(f"绘制完成，保存至：{save_path}")

    test_mAP_05 = evaluate_mAP(pred_boxes_list, gt_boxes_list, iou_thres=0.5)
    test_mAP_075 = evaluate_mAP(pred_boxes_list, gt_boxes_list, iou_thres=0.75)
    print(f'\nTest mAP@0.5: {test_mAP_05:.4f}')
    print(f'Test mAP@0.75: {test_mAP_075:.4f}')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    draw_original_car_labels()

    root_path = os.path.join('/', 'home', 'adduser', 'Shenji', 'UA-TRAC')
    img_root = os.path.join(root_path, 'Insight-MVT_Annotation_Train')
    original_gt_path = os.path.join(root_path, 'train_gt.txt')
    output_img_dir = create_dir(os.path.join(root_path, 'outputting_imgs'))
    new_gt_dir = create_dir(os.path.join(root_path, 'new_gt_files'))

    test_ratio = 0.1 
    val_ratio = 0.1
    epochs = 5
    batch_size = 8
    max_boxes = 30
    img_size = 640
    num_classes = 1
    max_samples = None

    gt_data = load_gt_data(original_gt_path)
    train_gt_path, val_gt_path, test_gt_path = shuffle_and_split_gt(gt_data, test_ratio, val_ratio, new_gt_dir)

    print("开始聚类锚框...")
    boxes_wh = load_boxes_from_gt(train_gt_path, img_root, img_size)
    anchors = kmeans_anchors(boxes_wh, n_clusters=9, img_size=img_size)
    print(f"聚类得到的锚框：{anchors}")
    anchors_tensor = torch.tensor(anchors, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    train_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=20.0, clip=True))

    val_test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=20.0, clip=True))

    train_dataset = UATracDatasetYOLO(train_gt_path, img_root, train_transform, num_classes, max_boxes, img_size, max_samples)
    val_dataset = UATracDatasetYOLO(val_gt_path, img_root, val_test_transform, num_classes, max_boxes, img_size, max_samples)
    test_dataset = UATracDatasetYOLO(test_gt_path, img_root, val_test_transform, num_classes, max_boxes, img_size, max_samples)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True, 
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=custom_collate_fn 
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    model = ResNetFPNYOLOv5(num_classes=num_classes, img_size=img_size, anchors=anchors_tensor).to(device)
    criterion = YOLOv5Loss(num_classes=num_classes, img_size=img_size, anchors=anchors_tensor).to(device)

    lr = 0.2
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    warmup_epochs = min(3, epochs)
    cosine_epochs = epochs - warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_epochs * len(train_loader)
    )
    if cosine_epochs > 0:
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cosine_epochs * len(train_loader),
            eta_min=lr * 1e-4
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * len(train_loader)]
        )
    else:
        scheduler = warmup_scheduler

    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, root_path, 
                        epochs=epochs, patience=5, grad_clip=5.0, accumulate_steps=2)
    test_model_with_draw(os.path.join(root_path, 'best_yolov5_resnet_fpn.pth'), test_loader, device, 
                         img_size=img_size, conf_thres=0.5, iou_thres=0.25, anchors=anchors_tensor)

    print(f"模型测试完成！最佳模型路径：{os.path.join(root_path, 'best_yolov5_resnet_fpn.pth')}")