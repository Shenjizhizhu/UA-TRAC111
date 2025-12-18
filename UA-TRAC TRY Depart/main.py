import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from config import *
from dataset import UATracDatasetYOLO, get_transforms
from model import ResNetFPNYOLOv5
from loss import YOLOv5Loss
from train import train_model, test_model_with_draw
from utils import *


if __name__ == '__main__':
    draw_original_car_labels()
    create_dir(OUTPUT_IMG_DIR)
    create_dir(NEW_GT_DIR)

    gt_data = load_gt_data(ORIGINAL_GT_PATH)
    train_gt_path, val_gt_path, test_gt_path = shuffle_and_split_gt(
        gt_data, TEST_RATIO, VAL_RATIO, NEW_GT_DIR
    )

    print("开始聚类锚框...")
    boxes_wh = load_boxes_from_gt(train_gt_path, IMG_ROOT, IMG_SIZE)
    anchors = kmeans_anchors(boxes_wh, n_clusters=9, img_size=IMG_SIZE)
    print(f"聚类得到的锚框：{anchors}")
    anchors_tensor = torch.tensor(anchors, device=DEVICE)

    train_transform, val_test_transform = get_transforms(IMG_SIZE)
    train_dataset = UATracDatasetYOLO(
        train_gt_path, IMG_ROOT, train_transform, NUM_CLASSES, MAX_BOXES, IMG_SIZE, MAX_SAMPLES
    )
    val_dataset = UATracDatasetYOLO(
        val_gt_path, IMG_ROOT, val_test_transform, NUM_CLASSES, MAX_BOXES, IMG_SIZE, MAX_SAMPLES
    )
    test_dataset = UATracDatasetYOLO(
        test_gt_path, IMG_ROOT, val_test_transform, NUM_CLASSES, MAX_BOXES, IMG_SIZE, MAX_SAMPLES
    )

    train_loader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
        collate_fn=custom_collate_fn
    )

    print(f"使用设备：{DEVICE}")
    model = ResNetFPNYOLOv5(NUM_CLASSES, IMG_SIZE, anchors_tensor).to(DEVICE)
    criterion = YOLOv5Loss(NUM_CLASSES, IMG_SIZE, anchors_tensor).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warmup_epochs = min(3, EPOCHS)
    cosine_epochs = EPOCHS - warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs * len(train_loader)
    )
    if cosine_epochs > 0:
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=cosine_epochs * len(train_loader), eta_min=LR * 1e-4
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * len(train_loader)]
        )
    else:
        scheduler = warmup_scheduler

    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, ROOT_PATH,
        EPOCHS, patience=5, grad_clip=GRAD_CLIP, accumulate_steps=ACCUMULATE_STEPS
    )

    test_model_with_draw(
        os.path.join(ROOT_PATH, 'best_yolov5_resnet_fpn.pth'), test_loader, DEVICE,
        IMG_SIZE, conf_thres=0.5, iou_thres=0.25, anchors=anchors_tensor
    )

    print(f"模型测试完成！最佳模型路径：{os.path.join(ROOT_PATH, 'best_yolov5_resnet_fpn.pth')}")