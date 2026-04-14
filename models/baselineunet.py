import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # Two Conv2d layers with ReLU
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU(),
                                  nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU()
                                  )
    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # Encoder: Conv block followed by MaxPool to downsample
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # skip connection output before pooling
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Decoder: ConvTranspose2d to upsample (deconvolution), then Conv block
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        )
        # After upsampling, concatenate skip connection by doubbling then conv
        self.conv = ConvBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        #first upsample, then concatenate skip connection then conv
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder which are successive Conv+Pool blocks, increasing channels
        # compressing spatial size
        self.e1 = EncoderBlock(3, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)

        # Bottleneck is the deepest compressed representation
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                        nn.ReLU())

        # Decoder which are  successive ConvTranspose+Conv blocks, decreasing channels
        # expanding spatial size
        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)

        self.out = nn.Sequential(nn.Conv2d(32, 2 , kernel_size=1))

    def forward(self, x):

        #encoder pass
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder pass
        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        # Output
        out = self.out(d3)

        return out

class UNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = "UNetClassifier"
        self.unet = UNet()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.unet_output = None

    def forward(self, x):
        x = self.unet(x)
        self.unet_output = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x