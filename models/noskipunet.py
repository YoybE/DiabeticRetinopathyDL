import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU(),
                                  nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # No skip connection: upsample then conv only (no concat, so in_c not doubled)
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c, out_c)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class NoSkipUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = EncoderBlock(3, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                        nn.ReLU())

        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)

        self.out = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        p1 = self.e1(x)
        p2 = self.e2(p1)
        p3 = self.e3(p2)

        b = self.bottleneck(p3)

        d1 = self.d1(b)
        d2 = self.d2(d1)
        d3 = self.d3(d2)

        return self.out(d3)

class NoSkipUNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = "NoSkipUNetClassifier"
        self.unet = NoSkipUNet()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.unet_output = None

    def forward(self, x):
        x = self.unet(x)
        self.unet_output = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
