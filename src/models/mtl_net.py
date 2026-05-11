import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class _SegDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


class _PoseDeconvHead(nn.Module):
    def __init__(self, in_ch: int, num_keypoints: int):
        super().__init__()
        self.deconvs = nn.Sequential(
            *[nn.Sequential(
                nn.ConvTranspose2d(in_ch if i == 0 else 256, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ) for i in range(3)]
        )
        self.head = nn.Conv2d(256, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.deconvs(x))


class HybridMTLNet(nn.Module):
    """ResNet-34 backbone with shared encoder, U-Net seg decoder, and deconv pose head.

    Outputs per-instance masks and heatmaps:
      seg  : (B, num_instances, H, W) raw logits  — BCEWithLogitsLoss, one channel per mouse
      pose : (B, num_keypoints,  H, W) heatmaps   — MSELoss, num_keypoints = 3 kp × num_instances
    """

    def __init__(self, num_instances: int = 2, num_keypoints: int = 6):
        super().__init__()
        base = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        layers = list(base.children())

        self.layer0 = nn.Sequential(*layers[:3])   # 64ch, /2
        self.layer1 = nn.Sequential(*layers[3:5])  # 64ch, /4
        self.layer2 = layers[5]                    # 128ch, /8
        self.layer3 = layers[6]                    # 256ch, /16
        self.layer4 = layers[7]                    # 512ch, /32

        self.seg_dec4 = _SegDecoderBlock(512, 256, 256)
        self.seg_dec3 = _SegDecoderBlock(256, 128, 128)
        self.seg_dec2 = _SegDecoderBlock(128, 64, 64)
        self.seg_dec1 = _SegDecoderBlock(64, 64, 64)
        self.seg_head = nn.Conv2d(64, num_instances, 1)

        self.pose_head = _PoseDeconvHead(512, num_keypoints)

    def forward(self, x: torch.Tensor):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        seg = self.seg_head(self.seg_dec1(self.seg_dec2(self.seg_dec3(self.seg_dec4(x4, x3), x2), x1), x0))
        seg = nn.functional.interpolate(seg, scale_factor=2, mode="bilinear", align_corners=True)

        pose = self.pose_head(x4)
        pose = nn.functional.interpolate(pose, size=seg.shape[2:], mode="bilinear", align_corners=True)

        return seg, pose
