import torch
import torch.nn as nn


def _multiresblock_splits(U, alpha=1.67):

    W = int(U * alpha)
    #number of filter for the 3 by 3 conv path
    nf3 = int(W / 3)
    nf7 = W - int(W * 2 / 3)
    nf5 = W - nf3 - nf7
    return nf3, nf5, nf7, W


class MultiResBlock(nn.Module):
    
    def __init__(self, U, in_c, alpha=1.67):
        super().__init__()
        nf3, nf5, nf7, W = _multiresblock_splits(U, alpha)

        # Each conv captures features at a progressively wider scale
        self.conv3 = nn.Sequential(nn.Conv2d(in_c, nf3, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(nf3), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(nf3, nf5, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(nf5), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(nf5, nf7, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(nf7), nn.ReLU())

        # Batch norm applied after concatenating all three outputs
        self.bn = nn.BatchNorm2d(W)

        # Shortcut projects the input to the same width as the concatenated output
        self.shortcut = nn.Sequential(nn.Conv2d(in_c, W, kernel_size=1),
                                      nn.BatchNorm2d(W))
        self.relu = nn.ReLU()
        self.out_c = W  # output channel count used when chaining blocks

    def forward(self, x):
        shortcut = self.shortcut(x)

        # Run the three sequential convs and collect each intermediate output
        a = self.conv3(x)   # 3x3 scale features
        b = self.conv5(a)   # 5x5 scale features (built on top of a)
        c = self.conv7(b)   # 7x7 scale features (built on top of b)

        # Combine all scales, then add the residual shortcut
        out = self.bn(torch.cat([a, b, c], dim=1))
        return self.relu(out + shortcut)


class ResPathBlock(nn.Module):
    """Single residual block used inside a ResPath."""
    def __init__(self, filters):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(filters))
        self.shortcut = nn.Sequential(nn.Conv2d(filters, filters, kernel_size=1),
                                      nn.BatchNorm2d(filters))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResPath(nn.Module):
    def __init__(self, filters, length):
        super().__init__()
        self.blocks = nn.ModuleList([ResPathBlock(filters) for _ in range(length)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class MultiResUNet(nn.Module):
    
    def __init__(self, alpha=1.67):
        super().__init__()
        U = [32, 64, 128, 256]  # equivalent UNet widths per level

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.mrb0 = MultiResBlock(U[0], 3, alpha)  # full resolution
        self.mrb1 = MultiResBlock(U[1], self.mrb0.out_c, alpha)  # 1/2 resolution
        self.mrb2 = MultiResBlock(U[2], self.mrb1.out_c, alpha)  # 1/4 resolution

        # Bottleneck
        self.mrb3 = MultiResBlock(U[3], self.mrb2.out_c, alpha)  # 1/8 resolution

        # ResPath skip connections — shallower levels get longer paths
        # because they have the largest semantic gap to bridge
        self.rp0 = ResPath(self.mrb0.out_c, length=3)
        self.rp1 = ResPath(self.mrb1.out_c, length=2)
        self.rp2 = ResPath(self.mrb2.out_c, length=1)

        # Decoder — each level upsamples then concatenates the ResPath-filtered skip
        self.dmrb2 = MultiResBlock(U[2], self.mrb3.out_c + self.mrb2.out_c, alpha)
        self.dmrb1 = MultiResBlock(U[1], self.dmrb2.out_c + self.mrb1.out_c, alpha)
        self.dmrb0 = MultiResBlock(U[0], self.dmrb1.out_c + self.mrb0.out_c, alpha)

        # 1x1 conv to produce 2-class output map
        self.out = nn.Conv2d(self.dmrb0.out_c, 2, kernel_size=1)

    def forward(self, x):
        # Encoder pass — extract multi-scale features at each resolution
        s0 = self.mrb0(x)
        s1 = self.mrb1(self.pool(s0))
        s2 = self.mrb2(self.pool(s1))
        b  = self.mrb3(self.pool(s2))  # bottleneck

        # Decoder pass — upsample, combine with ResPath-filtered skip, re-convolve
        d2 = self.dmrb2(torch.cat([self.up(b),  self.rp2(s2)], dim=1))
        d1 = self.dmrb1(torch.cat([self.up(d2), self.rp1(s1)], dim=1))
        d0 = self.dmrb0(torch.cat([self.up(d1), self.rp0(s0)], dim=1))

        return self.out(d0)


class MultiResUNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = "MultiResUNetClassifier"
        self.unet = MultiResUNet()
        # Collapse spatial dimensions to a single vector per image
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.unet_output = None

    def forward(self, x):
        x = self.unet(x)
        self.unet_output = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
