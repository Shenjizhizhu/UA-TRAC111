import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import load_gt_data


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


def get_transforms(img_size=640):
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

    return train_transform, val_test_transform




