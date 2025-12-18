import os
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import defaultdict


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
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
        gt_data, test_size=test_ratio, random_state=42, shuffle=True
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_ratio/(1-test_ratio), random_state=42, shuffle=True
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
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



