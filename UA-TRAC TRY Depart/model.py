import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


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