import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np 
import cv2 
from utils import evaluate_mAP, non_max_suppression, create_dir


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, root_path, 
                epochs=20, patience=5, grad_clip=5.0, accumulate_steps=2):
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


def test_model_with_draw(model_path, test_loader, device, img_size=640, conf_thres=0.5, iou_thres=0.25, anchors=None):
    from model import ResNetFPNYOLOv5
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