import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Two conv layers, each followed by batch norm and ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class NestedUNet(nn.Module):
    """
    UNet++ (Nested UNet) — an improved UNet where the decoder is redesigned
    to use dense skip connections instead of the single skip per level.
    """
    def __init__(self, filters=[32, 64, 128, 256]):
        super().__init__()
        f = filters
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder — same structure as a standard UNet
        self.x00 = ConvBlock(3,    f[0])  # full resolution
        self.x10 = ConvBlock(f[0], f[1])  # 1/2 resolution
        self.x20 = ConvBlock(f[1], f[2])  # 1/4 resolution
        self.x30 = ConvBlock(f[2], f[3])  # bottleneck at 1/8 resolution



        # First wave of intermediate nodes (j=1)
        self.x01 = ConvBlock(f[0] + f[1],   f[0])  
        self.x11 = ConvBlock(f[1] + f[2],   f[1])  
        self.x21 = ConvBlock(f[2] + f[3],   f[2])  

        # Second wave of intermediate nodes (j=2)
        self.x02 = ConvBlock(f[0]*2 + f[1], f[0])  
        self.x12 = ConvBlock(f[1]*2 + f[2], f[1])  

        # Final decoder node — combines all full-resolution features seen so far
        self.x03 = ConvBlock(f[0]*3 + f[1], f[0])  

        # 1x1 conv to produce 2-class output map
        self.out = nn.Conv2d(f[0], 2, kernel_size=1)

    def forward(self, x):
        # Encoder pass — compress spatial size, expand channels
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))  # bottleneck

        # Dense decode — build up intermediate nodes level by level.
        # Each node has access to more refined context than the last.
        x01 = self.x01(torch.cat([x00, self.up(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up(x20)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21)], dim=1))

        # Final output node collects all full-resolution features
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], dim=1))

        return self.out(x03)


class NestedUNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = "NestedUNetClassifier"
        self.unet = NestedUNet()
        # Collapse spatial dimensions to a single vector per image
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.unet_output = None

    def forward(self, x):
        x = self.unet(x)
        self.unet_output = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
