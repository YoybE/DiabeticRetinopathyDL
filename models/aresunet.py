import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # First Conv2d layer w/ BN
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c))        
        # Second Conv2d layer w/ BN
        self.conv2 = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c))
        
        self.relu = nn.ReLU()

        # Residual
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += shortcut
        x = self.relu(x)
        
        return x

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


    def forward(self, x, skip, attn):
        #first upsample, then concatenate skip connection then conv
        x = self.up(x)
        skip = skip*attn
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class AttentionGate(nn.Module):
    '''
    Attention Gates (AG) filter out features passed through skip connections
    '''
    def __init__(self, x_in_c, g_in_c, out_c):
        super().__init__()
        self.Wx_T = nn.Conv2d(x_in_c, out_c, kernel_size=1, stride=2) # x has lower amt of channels but higher resolution; requires stride of 2 to downsample
        self.Wg_T = nn.Conv2d(g_in_c, out_c, kernel_size=1, stride=1)
        self.psi = nn.ConvTranspose2d(out_c, out_c, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        '''
        x: skip connection
        g: gating signal
        '''

        # Projection so that x & g their feature resolutions match
        x_out = self.Wx_T(x)
        g_out = self.Wg_T(g)

        # Calculate the additive attention
        q = self.psi(self.relu(x_out + g_out))
        attn = self.sigmoid(q)

        return attn
    
class AttentionResUNet(nn.Module):
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

        # Attention layers introduced to end of skip connections
        self.a1 = AttentionGate(128, 256, 128)
        self.a2 = AttentionGate(64, 128, 64)
        self.a3 = AttentionGate(32, 64, 32)
        
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
        a1 = self.a1(x=s3,g=b)
        d1 = self.d1(b, skip=s3, attn=a1)
        a2 = self.a2(x=s2,g=d1)
        d2 = self.d2(d1, skip=s2, attn=a2)
        a3 = self.a3(x=s1,g=d2)
        d3 = self.d3(d2, skip=s1, attn=a3)

        # Output
        out = self.out(d3)

        return out
    
class AResUNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._name = "AResUNetClassifier"
        self.unet = AttentionResUNet()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.unet_output = None

    def forward(self, x):
        x = self.unet(x)
        self.unet_output = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x