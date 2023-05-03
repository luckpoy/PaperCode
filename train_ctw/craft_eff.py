import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("/home/tml/bo/CRAFT/CRAFT-Remade")
from src.effnetv2 import EffNetV2, effnetv2_s
from src.vgg16bn import init_weights
import train_ctw.config as cfg


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT_cct(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT_cct, self).__init__()

        """ Base network """
        self.basenet = effnetv2_s()

        """ U network """
        self.upconv1 = DoubleConv(256, 160, 128)
        self.upconv2 = DoubleConv(128, 64, 48)
        self.upconv3 = DoubleConv(48, 48, 24)
        self.upconv4 = DoubleConv(24, 24, 32)

        num_class = 1
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = sources[0]#256x24x24
        y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=False)#256x48x48
        y = torch.cat([y, sources[1]], dim=1)#416x48x48
        y = self.upconv1(y)#128x48x48

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)#128x96x96
        y = torch.cat([y, sources[2]], dim=1)#192x96x96
        y = self.upconv2(y)#48x96x96

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)#48x192x192
        y = torch.cat([y, sources[3]], dim=1)#96x192x192
        y = self.upconv3(y)#24x192x192

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)#24x384x384
        y = torch.cat([y, sources[4]], dim=1)#48x384x384
        feature = self.upconv4(y)#32x384x384

        y = self.conv_cls(feature)

        # ToDo - Remove the interpolation and make changes in the dataloader to make target width, height //2

        y = F.interpolate(y, size=(cfg.img_size, cfg.img_size), mode='bilinear', align_corners=False)

        return y

if __name__ == '__main__':
    model = CRAFT_cct().cuda()
    output = model(torch.randn(1, 3, 768, 768).cuda())
    print(output.shape)