import torch
import torch.nn as nn
import model.common as common
from einops import rearrange


class Encoder_lr(nn.Module):
    def __init__(self,feats = 64, scale=4):
        super(Encoder_lr, self).__init__()
        self.scale=scale
        self.E = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )
    def forward(self, x):
        x = rearrange(x, 'b u v c h w -> (b u v) c h w')
        # print("Encoder1:{}".format(x.device))
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1
    

class Encoder_gt(nn.Module):
    def __init__(self,feats = 64,scale=4):
        super(Encoder_gt, self).__init__()
        if scale == 2:
            in_dim = 15
        elif scale == 3:
            in_dim = 30
        elif scale == 4:
            in_dim = 51 ##51
        else :
            print('Upscale error!!!!')

        self.D = nn.Sequential(
            nn.Conv2d(in_dim, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )


        self.C = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=7, stride=7, padding=0),
            nn.LeakyReLU(0.1, True),            
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            DistgBlock(5, feats),
            # DistgBlock(5, feats),           
            nn.Conv2d(feats, feats*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(0.1, True),            
            nn.Conv2d(feats*2, feats*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(feats * 4 * 2, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
    def forward(self, x, gt):
        # print(x.shape)
        x = rearrange(x, 'b u v c h w -> (b u v) c h w')
        gt = rearrange(gt, 'b u v c h w -> (b u v) c h w')
        gt0 = self.pixel_unshuffle(gt)
        # print(x.shape)
        # print(gt0.shape) 
        x = torch.cat([x, gt0], dim=1)
        # print(x.shape)
        # print("Encoder:{}".format(x.device))
        x1_ave = self.D(x).squeeze(-1).squeeze(-1)
        x2_ave = self.C(gt).squeeze(-1).squeeze(-1)
        fea = self.mlp(torch.cat([x1_ave, x2_ave], dim=1))
        return fea

class denoise(nn.Module):
    def __init__(self,feats = 64, timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        self.mlp=nn.Sequential(
            nn.Linear(feats*4*2+1, feats*4),
            nn.LeakyReLU(0.1, True),           
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x, t, c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        fea = self.mlp(torch.cat([x,t,c],dim=1))
        return fea 

class DistgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DistgBlock, self).__init__()
        self.spa_conv = SpaConv(channels, channels)
        self.ang_conv = AngConv(angRes, channels, channels // 4)
        self.epi_conv = EpiConv(angRes, channels, channels // 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * channels + channels // 4, channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        
        x1 = rearrange(x, '(b u v) c h w -> b u v c h w',u=5,v=5)
        b, u, v, c, h, w = x1.shape
        fea_spa = self.spa_conv(x1)
        fea_ang = self.ang_conv(x1)
        fea_epih = self.epi_conv(x1)
        xT = rearrange(x1, 'b u v c h w -> b v u c w h')
        fea_epiv = rearrange(self.epi_conv(xT), 'b v u c w h -> b u v c h w')
        fea = torch.cat((fea_spa, fea_ang, fea_epih, fea_epiv), dim=3)
        fea = rearrange(fea, 'b u v c h w -> (b u v) c h w')
        out = self.fuse(fea)
        # out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out + x


class SpaConv(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(SpaConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, True))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input = rearrange(x, 'b u v c h w -> (b u v) c h w')
        out = self.conv(input)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        return out


class AngConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super(AngConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes * angRes * channel_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.PixelShuffle(angRes))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input_ang = rearrange(x, 'b u v c h w -> (b h w) c u v')
        out = self.conv(input_ang)
        out = rearrange(out, '(b h w) c u v -> b u v c h w', b=b, h=h, w=w)

        return out


class EpiConv(nn.Module):
    def __init__(self, angRes, channel_in, channel_out):
        super(EpiConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=angRes, stride=1, padding=(0, angRes//2), bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channel_out, angRes * channel_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, True),
            PixelShuffle1D(angRes))

    def forward(self, x):
        b, u, v, c, h, w = x.shape
        input_epi = rearrange(x, 'b u v c h w -> (b u h) c v w')
        out = self.conv(input_epi)
        out = rearrange(out, '(b u h) c v w -> b u v c h w', b=b, u=u, h=h)

        return out
    



class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor

        return x.view(b, c, h * self.factor, w)