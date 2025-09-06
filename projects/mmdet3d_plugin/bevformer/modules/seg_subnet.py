import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18, resnet50
from ..modules.builder import SEG_ENCODER

from mmcv.runner import BaseModule
from mmseg.models.builder import HEADS
import torch.nn.functional as F
#from mmseg.models.decode_heads.segformer_head import SegFormerHead


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1) #相当与通道维度上连接，以弥补因为使用mb导致的卷积信息丢失。
        return self.conv(x1)

# ----------------------------
# UNet3Res18
# ----------------------------
@SEG_ENCODER.register_module()
class UNet3Res18(nn.Module):
    """
    2 downs → 1 up:
      downs: conv1(stride2) → H/2, layer2(stride2) → H/4
      up  : fuse layer2 & layer1 back to H/2 → final conv to H
    Input: [B,256,200,400]
    """
    def __init__(self, inC, outC):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)

        # --- Encoder (2 downs) ---
        # conv1: [B,256,200,400] → [B, 64,100,200]
        self.conv1 = nn.Conv2d(inC,  64, 7, stride=2, padding=3, bias=False)
        self.bn1   = trunk.bn1
        self.relu  = trunk.relu

        # layer1 (no down): [B, 64,100,200] → [B, 64,100,200]
        self.layer1 = trunk.layer1

        # layer2 (2nd down): [B, 64,100,200] → [B,128, 50,100]
        self.layer2 = trunk.layer2

        # --- Decoder (1 up) ---
        # fuse layer2 (128ch@50×100) with layer1 (64ch@100×200)
        # we upsample layer2 by ×2 → 100×200, concat → 128+64=192ch, output 64ch
        self.up1 = Up(128 + 64, 64, scale_factor=2)  # → [B,64,100,200]

        # final 1×1 to map 64 → outC, then upsample to full resolution
        self.classifier = nn.Conv2d(64, outC, kernel_size=1)  # → [B,outC,100,200]
        self.final_up   = nn.Upsample(scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True)       # → [B,outC,200,400]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,256,200,400]
        x0 = self.relu(self.bn1(self.conv1(x)))  # [B, 64,100,200]
        x1 = self.layer1(x0)                     # [B, 64,100,200]
        x2 = self.layer2(x1)                     # [B,128, 50,100]

        y  = self.up1(x2, x1)                    # [B, 64,100,200]
        y  = self.classifier(y)                  # [B,outC,100,200]
        return self.final_up(y)                  # [B,outC,200,400]

# ----------------------------
# UNet4Res18
# ----------------------------
@SEG_ENCODER.register_module()
class UNet4Res18(nn.Module):
    def __init__(self, inC, outC):
        super(UNet4Res18, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x): #torch.Size([B, 256, 200, 400])
        x = self.conv1(x) #torch.Size([B, 64, 100, 200])
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x) #torch.Size([B, 64, 100, 200])
        x = self.layer2(x1) #torch.Size([B, 128, 50, 100])
        x2 = self.layer3(x) #torch.Size([B, 256, 25, 50])

        x = self.up1(x2, x1) #torch.Size([B, 256, 100, 200])
        x = self.up2(x) #torch.Size([B, 4, 200, 400]) 语义分割预测特征图

        return x
    
# ----------------------------
# UNet5Res18
# ----------------------------
@SEG_ENCODER.register_module()
class UNet5Res18(nn.Module):
    """
    4-down / 3-up U-Net with an inner Up class (same name) that
    does dynamic, perfectly aligned upsamples.
    Input: [B,256,200,400]
    """

    # inner Up shadows global Up
    class Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x_low, x_skip):
            # upsample exactly to skip’s size
            x = F.interpolate(
                x_low, size=x_skip.shape[2:],
                mode='bilinear', align_corners=True
            )
            x = torch.cat([x_skip, x], dim=1)
            return self.conv(x)

    def __init__(self, inC, outC):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)

        # Encoder
        self.conv1  = nn.Conv2d(inC,  64, 7, stride=2, padding=3, bias=False)  # → [B,64,100,200]
        self.bn1    = trunk.bn1
        self.relu   = trunk.relu
        self.layer1 = trunk.layer1    # → [B,64,100,200]
        self.layer2 = trunk.layer2    # → [B,128, 50,100]
        self.layer3 = trunk.layer3    # → [B,256, 25, 50]
        self.layer4 = trunk.layer4    # → [B,512, 13, 25]

        # Decoder using inner Up
        self.up1 = UNet5Res18.Up(512 + 256, 256)  # fuse x4 & x3 → [B,256,25,50]
        self.up2 = UNet5Res18.Up(256 + 128, 128)  # fuse u1 & x2 → [B,128,50,100]
        self.up3 = UNet5Res18.Up(128 +  64,  64)  # fuse u2 & x1 → [B, 64,100,200]

        # Final head + upsample to full res
        self.classifier = nn.Conv2d(64, outC, kernel_size=1)  # → [B,outC,100,200]
        self.final_up   = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )  # → [B,outC,200,400]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,256,200,400]
        x0 = self.relu(self.bn1(self.conv1(x)))    # → [B,64,100,200]
        x1 = self.layer1(x0)                       # → [B,64,100,200]
        x2 = self.layer2(x1)                       # → [B,128,50,100]
        x3 = self.layer3(x2)                       # → [B,256,25,50]
        x4 = self.layer4(x3)                       # → [B,512,13,25]

        u1 = self.up1(x4, x3)  # → [B,256,25,50]
        u2 = self.up2(u1, x2)  # → [B,128,50,100]
        u3 = self.up3(u2, x1)  # → [B, 64,100,200]

        seg = self.classifier(u3)  # → [B,outC,100,200]
        return self.final_up(seg)  # → [B,outC,200,400]
    
@SEG_ENCODER.register_module()
class UNet5Res50(nn.Module):
    """
    4-down / 3-up U-Net using ResNet-50 backbone.
    Follows original UNet5Res18 design but replaces ResNet-18 with ResNet-50.
    Input: [B,256,200,400]
    """
    class Up(nn.Module):
        """Inner Up block: two 3×3 conv layers"""
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x_low, x_skip):
            x = F.interpolate(x_low, size=x_skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x_skip, x], dim=1)
            return self.conv(x)

    def __init__(self, inC, outC):
        super().__init__()
        trunk = resnet50(pretrained=False, zero_init_residual=True)
        # Encoder
        self.conv1  = nn.Conv2d(inC, 64, 7, stride=2, padding=3, bias=False)  # → [B,64,100,200]
        self.bn1    = trunk.bn1
        self.relu   = trunk.relu
        self.layer1 = trunk.layer1    # → [B,256,100,200]
        self.layer2 = trunk.layer2    # → [B,512,50,100]
        self.layer3 = trunk.layer3    # → [B,1024,25,50]
        self.layer4 = trunk.layer4    # → [B,2048,13,25]
        # Decoder
        self.up1 = UNet5Res50.Up(2048 + 1024, 1024)  # → [B,1024,25,50]
        self.up2 = UNet5Res50.Up(1024 + 512, 512)    # → [B,512,50,100]
        self.up3 = UNet5Res50.Up(512 + 256, 256)     # → [B,256,100,200]
        # Final head + upsample
        self.classifier = nn.Conv2d(256, outC, kernel_size=1)  # → [B,outC,100,200]
        self.final_up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # → [B,outC,200,400]
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B,256,200,400]
        x0 = self.relu(self.bn1(self.conv1(x)))   # → [B,64,100,200]
        x1 = self.layer1(x0)                      # → [B,256,100,200]
        x2 = self.layer2(x1)                      # → [B,512,50,100]
        x3 = self.layer3(x2)                      # → [B,1024,25,50]
        x4 = self.layer4(x3)                      # → [B,2048,13,25]

        u1 = self.up1(x4, x3)                     # → [B,1024,25,50]
        u2 = self.up2(u1, x2)                     # → [B,512,50,100]
        u3 = self.up3(u2, x1)                     # → [B,256,100,200]

        seg = self.classifier(u3)                 # → [B,outC,100,200]
        return self.final_up(seg)                 # → [B,outC,200,400]
    
@SEG_ENCODER.register_module()
class UNet6Res18(UNet5Res18):
    """
    Extend UNet5Res18 by adding one extra down/up stage at the deepest level.
    Only the deepest layer (down0/up4) differs; all other logic inherited.
    """
    class Up(UNet5Res18.Up):
        # reuse same double-conv block
        pass

    def __init__(self, inC, outC):
        super().__init__(inC, outC)
        # Extra down0 stage before conv1
        self.conv0 = nn.Sequential(
            nn.Conv2d(inC, 64, 3, stride=2, padding=1, bias=False),  # → [B,64,100,200]
            nn.BatchNorm2d(64),                                         # → [B,64,100,200]
            nn.ReLU(inplace=True),                                      # relu
        )
        # Override conv1 to take 64→64 (was 256→64 in UNet4)
        self.conv1 = nn.Conv2d(64, 64, 7, stride=2, padding=3, bias=False)  # → [B,64,50,100]
        # New deepest Up4: fuse u3 & x0
        self.up4 = UNet6Res18.Up(64 + 64, 64)     # u3 & x0 → [B,64,100,200]

    def forward(self, x):
        # x: [B,256,200,400]
        x0 = self.conv0(x)                          # → [B,64,100,200]
        # then original conv1 bn1 relu from UNet4 but on x0
        x1 = self.relu(self.bn1(self.conv1(x0)))    # → [B,64,50,100]
        x1 = self.layer1(x1)                        # → [B,64,50,100]
        x2 = self.layer2(x1)                        # → [B,128,25,50]
        x3 = self.layer3(x2)                        # → [B,256,13,25]
        x4 = self.layer4(x3)                        # → [B,512,13,25]

        u1 = self.up1(x4, x3)  # → [B,256,25,50]
        u2 = self.up2(u1, x2)  # → [B,128,50,100]
        u3 = self.up3(u2, x1)  # → [B,64,100,200]
        u4 = self.up4(u3, x0)  # → [B,64,100,200]

        seg = self.classifier(u4)                  # → [B,outC,100,200]
        return self.final_up(seg)                  # → [B,outC,200,400]


@SEG_ENCODER.register_module()
class Conv2(nn.Module):

    def __init__(self, inC, outC):
        super(Conv2, self).__init__()
        self.seg_head = nn.Sequential(
            nn.Conv2d(inC, inC, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inC, outC , kernel_size=1))

    def forward(self, x):
        return self.seg_head(x)
    
@SEG_ENCODER.register_module()
class Linear2(nn.Module):

    def __init__(self, inC, outC, hidden_dim=None):
        """
        inC: number of input channels
        outC: number of output (semantic) channels / classes
        hidden_dim: internal MLP width (defaults to inC if None)
        """
        super(Linear2, self).__init__()
        hidden_dim = hidden_dim or inC
        self.mlp = nn.Sequential(
            nn.Linear(inC, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, outC)
        )

        # Initialize weights right after construction
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B, inC, H, W]
        returns: [B, outC, H, W]
        """
        B, C, H, W = x.shape
        # move channels last and flatten spatial dims
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)     # [B, H*W, inC]
        out_flat = self.mlp(x_flat)                             # [B, H*W, outC]
        # reshape back to [B, outC, H, W]
        out = out_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, outC, H, W]
        return out


import math
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16


@SEG_ENCODER.register_module()
class DeconvEncode(BaseModule):
    """The neck used in `CenterNet <https://arxiv.org/abs/1904.07850>`_ for
    object classification and box regression.

    Args:
         in_channel (int): Number of input channels.
         num_deconv_filters (tuple[int]): Number of filters per stage.
         num_deconv_kernels (tuple[int]): Number of kernels per stage.
         use_dcn (bool): If True, use DCNv2. Default: True.
         init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channel,
                 num_deconv_filters,
                 num_deconv_kernels,
                 outC=4,
                 use_dcn=True,
                 init_cfg=None):
        super(DeconvEncode, self).__init__(init_cfg)
        assert len(num_deconv_filters) == len(num_deconv_kernels)
        self.fp16_enabled = False
        self.use_dcn = use_dcn
        self.in_channel = in_channel
        self.deconv_layers = self._make_deconv_layer(num_deconv_filters,
                                                     num_deconv_kernels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(num_deconv_filters[-1], num_deconv_filters[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_deconv_filters[-1], outC , kernel_size=1))

    def _make_deconv_layer(self, num_deconv_filters, num_deconv_kernels):
        """use deconv layers to upsample backbone's output."""
        layers = []
        for i in range(len(num_deconv_filters)):
            feat_channel = num_deconv_filters[i]
            conv_module = ConvModule(
                self.in_channel,
                feat_channel,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=dict(type='BN'))
            layers.append(conv_module)
            upsample_module = ConvModule(
                feat_channel,
                feat_channel,
                num_deconv_kernels[i],
                stride=2,
                padding=1,
                conv_cfg=dict(type='deconv'),
                norm_cfg=dict(type='BN'))
            layers.append(upsample_module)
            self.in_channel = feat_channel

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (
                                    1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # self.use_dcn is False
            elif not self.use_dcn and isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    @auto_fp16()
    def forward(self, inputs):
        outs = self.deconv_layers(inputs)
        outs = self.seg_head(outs)
        return outs

"""

dict(
type='DeconvEncode',
in_channel=256,
num_deconv_filters=(256, 128, 64),
num_deconv_kernels=(4, 4, 4),
use_dcn=True),
        
"""

# ----------------------------
#  FPN-Style Multi-Scale Decoder
# ----------------------------
@SEG_ENCODER.register_module()
class FPN1(nn.Module):
    """
    Single-scale version: no in_channels_list needed.
    """
    def __init__(self, inC, outC, fuse_channels=64):
        super().__init__()
        # lateral conv (1x1) to fuse_channels
        self.lateral = nn.Conv2d(inC, fuse_channels, kernel_size=1)
        # head to go from fuse_channels -> outC
        self.fuse_head = nn.Sequential(
            nn.Conv2d(fuse_channels, fuse_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_channels, outC, kernel_size=1)
        )
        self.init_weights()

    def init_weights(self):
        # lateral
        nn.init.kaiming_normal_(self.lateral.weight, mode='fan_out', nonlinearity='relu')
        if self.lateral.bias is not None:
            nn.init.zeros_(self.lateral.bias)
        # fuse head
        for m in self.fuse_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat):
        """
        Args:
            feat (Tensor): [B, inC, H, W]
        Returns:
            Tensor: [B, outC, H, W]
        """
        x = self.lateral(feat)
        return self.fuse_head(x)

#   Input Feature Map
#       ├── Lateral 1x1 conv (base res)
#       ├── Downsample (MaxPool)
#       │    └── Lateral 1x1 conv
#       └── Upsample back to base res
#       ↓
#   Concatenate both branches
#       ↓
#   Fuse with conv → BN → ReLU
#       ↓
#   Final 1x1 conv → output logits
@SEG_ENCODER.register_module()
class FPN2Conv(nn.Module):
    """
    FPN-style decoder with 2-layer depth:
    - One base resolution (H, W)
    - One downsampled stage (H/2, W/2), upsampled and fused back
    """
    def __init__(self, inC, outC, fuse_channels=64):
        super().__init__()
        # Stage 1: base resolution lateral
        self.lateral1 = nn.Conv2d(inC, fuse_channels, kernel_size=1)

        # Stage 2: downsample + lateral
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lateral2 = nn.Conv2d(inC, fuse_channels, kernel_size=1)
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Fuse both stages
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True)
        )

        # Output head
        self.head = nn.Conv2d(fuse_channels, outC, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat):
        """
        Args:
            feat (Tensor): [B, inC, H, W]
        Returns:
            Tensor: [B, outC, H, W]
        """
        x1 = self.lateral1(feat)                        # [B, C, H, W]
        x2_down = self.downsample(feat)                # [B, C, H/2, W/2]
        x2 = self.lateral2(x2_down)                    # [B, C, H/2, W/2]
        x2_up = self.upconv2(x2)                       # [B, C, H, W]
        x_fused = torch.cat([x1, x2_up], dim=1)        # [B, 2C, H, W]
        x_fused = self.fuse(x_fused)                   # [B, C, H, W]
        return self.head(x_fused)                      # [B, outC, H, W]
    
@SEG_ENCODER.register_module()
class FPN3Conv(nn.Module):
    """
    FPN-style decoder with 3-layer depth:
    - Base resolution (H, W)
    - Downsampled (H/2, W/2)
    - Further downsampled (H/4, W/4)
    """
    def __init__(self, inC, outC, fuse_channels=64):
        super().__init__()
        # Stage 1: base resolution lateral
        self.lateral1 = nn.Conv2d(inC, fuse_channels, kernel_size=1)

        # Stage 2: downsample + lateral + upsample
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lateral2 = nn.Conv2d(inC, fuse_channels, kernel_size=1)
        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Stage 3: further downsample + lateral + upsample to base
        # reuse downsample twice
        self.lateral3 = nn.Conv2d(inC, fuse_channels, kernel_size=1)
        self.upconv3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Fuse all three stages
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels * 3, fuse_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True)
        )

        # Output head
        self.head = nn.Conv2d(fuse_channels, outC, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat):
        # Base resolution
        x1 = self.lateral1(feat)                # [B, C, H, W]
        # Stage 2
        x2_down = self.downsample(feat)         # [B, C, H/2, W/2]
        x2 = self.lateral2(x2_down)             # [B, C, H/2, W/2]
        x2_up = self.upconv2(x2)                # [B, C, H, W]
        # Stage 3
        x3_down = self.downsample(x2_down)      # [B, C, H/4, W/4]
        x3 = self.lateral3(x3_down)             # [B, C, H/4, W/4]
        x3_up = self.upconv3(x3)                # [B, C, H, W]
        # Fuse and head
        x_cat = torch.cat([x1, x2_up, x3_up], dim=1)  # [B, 3C, H, W]
        x_fused = self.fuse(x_cat)                    # [B, C, H, W]
        return self.head(x_fused)                    # [B, outC, H, W]


# ----------------------------
# FPN5Res18 (UNet-like simplicity, FPN fusion)
# ----------------------------
@SEG_ENCODER.register_module()
class FPN5Res18(nn.Module):
    """
    Simple ResNet18 + FPN decoder:
      - Stem: conv1/bn1/relu (NO maxpool)
      - Features: layer1..4 (C2..C5)
      - Laterals: 1x1 to 'fuse_channels'
      - Upsample all to C2 size, concat, fuse -> head
      - Final upsample back to input resolution

    Args:
        inC (int): input channels (e.g., BEV feature channels)
        outC (int): output channels (e.g., #seg classes)
        fuse_channels (int): channels used inside FPN fusion (default: 64)
    """
    def __init__(self, inC: int, outC: int, fuse_channels: int = 64):
        super().__init__()

        # ResNet-18 trunk
        trunk = resnet18(pretrained=False, zero_init_residual=True)

        # ---- Stem (match your UNet4Res18: conv1 stride=2, but NO maxpool) ----
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = trunk.bn1
        self.relu  = trunk.relu
        # DO NOT add trunk.maxpool here

        # ---- Backbone stages ----
        self.layer1 = trunk.layer1  # C2:  64ch, stride 2 (relative to input)
        self.layer2 = trunk.layer2  # C3: 128ch, stride 4
        self.layer3 = trunk.layer3  # C4: 256ch, stride 8
        self.layer4 = trunk.layer4  # C5: 512ch, stride 16

        # ---- 1x1 Laterals -> fuse_channels ----
        self.lat2 = nn.Conv2d( 64, fuse_channels, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(128, fuse_channels, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(256, fuse_channels, kernel_size=1, bias=False)
        self.lat5 = nn.Conv2d(512, fuse_channels, kernel_size=1, bias=False)

        # ---- Fuse (concat) ----
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels * 4, fuse_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True),
        )

        # ---- Head ----
        self.head = nn.Conv2d(fuse_channels, outC, kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x:  [B, inC, H, W]  (e.g., [B, 256, 200, 400])
        out:[B, outC, H, W]
        """
        B, _, H, W = x.shape

        # ---- Stem ----
        x = self.conv1(x)   # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)

        # ---- Pyramid features ----
        c2 = self.layer1(x)      # [B,  64, H/2,  W/2]
        c3 = self.layer2(c2)     # [B, 128, H/4,  W/4]
        c4 = self.layer3(c3)     # [B, 256, H/8,  W/8]
        c5 = self.layer4(c4)     # [B, 512, H/16, W/16]

        # ---- Laterals ----
        p2 = self.lat2(c2)  # target resolution for fusion
        target_hw = p2.shape[-2:]

        # Always upsample by explicit size to avoid rounding mismatches
        p3 = F.interpolate(self.lat3(c3), size=target_hw, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.lat4(c4), size=target_hw, mode='bilinear', align_corners=False)
        p5 = F.interpolate(self.lat5(c5), size=target_hw, mode='bilinear', align_corners=False)

        # ---- Concat & fuse ----
        x = torch.cat([p2, p3, p4, p5], dim=1)    # [B, 4*fuseC, H/2, W/2]
        x = self.fuse(x)                          # [B, fuseC,  H/2, W/2]

        # ---- Head + up to input size ----
        x = self.head(x)                          # [B, outC, H/2, W/2]
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x

@SEG_ENCODER.register_module()
class FPN(nn.Module):
    """
    Flexible FPN-style decoder with customizable depth levels.
    Supports levels from 2 to 7:
      - Level 1: original resolution (H, W)
      - Level 2: downsampled once (H/2, W/2)
      - ...
      - Level N: downsampled N-1 times (H/2^(N-1), W/2^(N-1))
    Then each level is upsampled back to original (H, W) and fused.
    """
    def __init__(self, inC, outC, fuse_channels=64, levels=4):
        super().__init__()
        assert 2 <= levels <= 7, "Levels must be between 2 and 7"
        self.levels = levels

        # Create lateral 1x1 convs for each level
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(inC, fuse_channels, kernel_size=1)
            for _ in range(levels)
        ])

        # Downsample module (used iteratively)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fusion head: combine concat of all levels
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels * levels, fuse_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True)
        )

        # Final classification head
        self.head = nn.Conv2d(fuse_channels, outC, kernel_size=1)

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize Conv2d and Norm layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat):
        """
        feat: [B, inC, H, W]
        Returns: [B, outC, H, W]
        """
        # Collect features at each level by iterative downsampling
        feats = [feat]
        for _ in range(self.levels - 1):
            feats.append(self.downsample(feats[-1]))

        # Apply lateral convs
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        # Upsample all levels back to original spatial size
        target_size = feat.shape[2:]
        upsampled = []
        for lvl, lat in enumerate(laterals):
            if lvl == 0:
                upsampled.append(lat)
            else:
                upsampled.append(
                    F.interpolate(lat, size=target_size, mode='bilinear', align_corners=False)
                )

        # Fuse and classify
        x = torch.cat(upsampled, dim=1)
        x = self.fuse(x)
        return self.head(x)

@SEG_ENCODER.register_module()
class FPN7Conv(FPN):
    """
    FPN-style decoder fixed at 7 levels (resolutions: H, H/2, H/4, H/8, H/16, H/32, H/64).
    Each level is upsampled back to the original (H, W) and fused.
    """
    def __init__(self, inC, outC, fuse_channels=64):
        # instantiate base FPN with levels=7
        super().__init__(inC, outC, fuse_channels, levels=7)



# ----------------------------
# FPN4Res18 (3-level pyramid: C2, C3, C4)
# ----------------------------
@SEG_ENCODER.register_module()
class FPN4Res18(nn.Module):
    """
    ResNet-18 encoder with a 3-level FPN fusion (C2..C4):
      c2: layer1 -> [B,  64, H/2,  W/2]
      c3: layer2 -> [B, 128, H/4,  W/4]
      c4: layer3 -> [B, 256, H/8,  W/8]

    FPN fusion:
      1x1 laterals to fuse_channels, upsample c3/c4 to c2 size, concat(3), fuse, head, upsample to input.
    """
    def __init__(self, inC: int, outC: int, fuse_channels: int = 64):
        super().__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)

        # Stem (no maxpool to match your UNet variants)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = trunk.bn1
        self.relu  = trunk.relu

        # Use only layer1..layer3 (C2..C4)
        self.layer1 = trunk.layer1   # 64ch, H/2
        self.layer2 = trunk.layer2   # 128ch, H/4
        self.layer3 = trunk.layer3   # 256ch, H/8

        # Laterals (1x1 -> fuse_channels)
        self.lat2 = nn.Conv2d( 64, fuse_channels, kernel_size=1, bias=False)  # C2
        self.lat3 = nn.Conv2d(128, fuse_channels, kernel_size=1, bias=False)  # C3
        self.lat4 = nn.Conv2d(256, fuse_channels, kernel_size=1, bias=False)  # C4

        # Fuse (concat 3)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels * 3, fuse_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_channels),
            nn.ReLU(inplace=True),
        )

        # Head
        self.head = nn.Conv2d(fuse_channels, outC, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x:  [B, inC, H, W]  e.g. [B,256,200,400]
        out:[B, outC, H, W]
        """
        B, _, H, W = x.shape

        # Encoder taps
        x  = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2,  W/2]
        c2 = self.layer1(x)                      # [B, 64, H/2,  W/2]
        c3 = self.layer2(c2)                     # [B,128, H/4,  W/4]
        c4 = self.layer3(c3)                     # [B,256, H/8,  W/8]

        # Laterals + upsample to c2 size
        p2 = self.lat2(c2)
        target_hw = p2.shape[-2:]
        p3 = F.interpolate(self.lat3(c3), size=target_hw, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.lat4(c4), size=target_hw, mode='bilinear', align_corners=False)

        # Concat & fuse
        f = torch.cat([p2, p3, p4], dim=1)       # [B, 3*fuseC, H/2, W/2]
        f = self.fuse(f)                          # [B,   fuseC, H/2, W/2]

        # Head + up to input size
        seg = self.head(f)                        # [B, outC, H/2, W/2]
        seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)
        return seg

# ----------------------------
#  DeepLabV3+ Style Decoder (ASPP + Conv Head)
#  Input BEV feature map (from encoder)
#          │
#       [ASPP]
#    ┌────┼────────────────────────────────────┐
#    │    ├── 1x1 conv (no dilation)           │
#    │    ├── 3x3 conv (d=6)                   │
#    │    ├── 3x3 conv (d=12)                  │
#    │    ├── 3x3 conv (d=18)                  │
#    │    └── image-level pooling + upsample  ─┘
#         ↓
#     Concatenation → 1x1 conv projection
#         ↓
#       [Head]
#         ↓
#     Final semantic mask (logits)
# ----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(self.branches) + 1) * out_channels,
                      out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        res = [b(x) for b in self.branches]
        img_feat = self.image_pool(x)
        img_feat = F.interpolate(img_feat, size=x.shape[2:],
                                  mode='bilinear', align_corners=False)
        res.append(img_feat)
        x = torch.cat(res, dim=1)
        return self.project(x)

# ----------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for d in dilations
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(self.branches) + 1) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        res = [b(x) for b in self.branches]
        img = self.image_pool(x)
        img = F.interpolate(img, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(img)
        return self.project(torch.cat(res, dim=1))


@SEG_ENCODER.register_module()
class DeepLabV3Plus(nn.Module):
    def __init__(self, inC, outC, aspp_out=256):
        super().__init__()
        self.aspp = ASPP(inC, aspp_out)
        self.head = nn.Sequential(
            nn.Conv2d(aspp_out, aspp_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(aspp_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(aspp_out, outC, 1)
        )
        self.init_weights()

    def init_weights(self):
        # ASPP
        for m in self.aspp.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # head
        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat):
        x = self.aspp(feat)
        return self.head(x)


class SegFormerHead(nn.Module):
    """
    Simplified SegFormer decode head:
      - Projects multi-scale inputs via 1×1 convs.
      - Upsamples to the highest resolution.
      - Concatenates, fuses, and classifies.
    """
    def __init__(self, in_channels_list, embed_dim=256, num_classes=80, dropout_ratio=0.1):
        super().__init__()
        self.num_stages = len(in_channels_list)
        # per-stage projection
        self.proj_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for in_ch in in_channels_list
        ])
        # fusion & dropout
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_stages, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio)
        )
        # classifier
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feats):
        # feats: list of [B, C_i, H_i, W_i]
        h, w = feats[0].shape[2:]
        outs = []
        for f, proj in zip(feats, self.proj_convs):
            x = proj(f)
            if x.shape[2:] != (h, w):
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        return self.classifier(x)


@SEG_ENCODER.register_module()
class PanopticSegFormerDecoder(BaseModule):
    """
    Panoptic SegFormer decoder for BEV feature maps,
    self-contained (no external mmseg dependency).
    """
    def __init__(self,
                 inC: int,
                 outC: int,
                 channels: int = 256,
                 num_stages: int = 4,
                 dropout_ratio: float = 0.1,
                 init_cfg: dict = None):
        super().__init__(init_cfg)
        # Build a SegFormerHead expecting `num_stages` inputs of `inC` channels
        self.segformer = SegFormerHead(
            in_channels_list=[inC] * num_stages,
            embed_dim=channels,
            num_classes=outC,
            dropout_ratio=dropout_ratio
        )
        self.init_weights()

    def init_weights(self):
        # Delegate to SegFormerHead init
        if hasattr(self.segformer, 'init_weights'):
            self.segformer.init_weights()
        else:
            # fallback init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, bev_feats, img_metas=None, **kwargs):
        # Replicate single feature map if needed
        if not isinstance(bev_feats, (list, tuple)):
            bev_feats = [bev_feats] * self.segformer.num_stages
        return self.segformer(bev_feats)