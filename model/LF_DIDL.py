import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb, LF_interpolate
from einops import rearrange
from model.diffusion import diffusion
from model.encoder_SAE import Encoder_lr, Encoder_gt, denoise
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import numbers


class LF_DIDL(nn.Module):
    def __init__(self, args):
        super(LF_DIDL, self).__init__()
        timesteps = 8 # default is 4
        self.angRes = args.angRes
        self.scale = args.upfactor
        self.SR = LFnet(args).cuda()
        self.encoder = Encoder_gt(feats=64, scale=args.upfactor).cuda()
        self.condition = Encoder_lr(feats=64, scale=args.upfactor).cuda()
        self.denoise = denoise(feats=64, timesteps=8).cuda()
        self.netG = diffusion.DDPM(denoise=self.denoise, 
        condition=self.condition ,feats=64, timesteps = timesteps
        ).cuda()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.training:
            img = x[0]
            gt = x[1]
            diffusion = x[2]
            if diffusion == False:      
                # encode CDP
                cdp = self.encoder(img,gt)
                cdp_diff = cdp
                sr = self.SR(img, cdp)
            else:
                # freezen CDP
                self.freeze_module(self.encoder) 
                # encode CDP
                cdp = self.encoder(img,gt)
                # generate CDP
                cdp_diff = self.netG(img, cdp)
                sr = self.SR(img, cdp_diff)
            return cdp_diff, cdp, sr
        else:
            img = x[0]
            gt = x[1]
            diffusion = x[2]
            if diffusion == False:      
                # encode CDP
                cdp = self.encoder(img,gt)
                sr = self.SR(img, cdp)
            else:
                # generate CDP
                cdp_diff = self.netG(img)    
                sr = self.SR(img, cdp_diff) 
            return sr




class LFnet(nn.Module):
    def __init__(self, args):
        super(LFnet, self).__init__()
        channels = 64
        self.angRes = args.angRes
        self.scale = args.upfactor

        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(3, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        # self.altblock = AltFilter(in_channels=channels, connection="uv")


        self.altblock = nn.ModuleList([
            AltFilter(in_channels=channels, connection="uv") for _ in range(14)
        ])
            
            
        # self.altblock = nn.Sequential(
        #     AltFilter(self.angRes, channels),
        #     # AltFilter(self.angRes, channels),
        #     # AltFilter(self.angRes, channels),
        #     # AltFilter(self.angRes, channels),
        #     # AltFilter(self.angRes, channels),
        # )

        ########################### UP-Sampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr, cdp):
        lr = rearrange(lr, 'b u v c h w -> b c u v h w')

        [b, c, u, v, h, w] = lr.size()

        # Initial Feature Extraction
        x = rearrange(lr, 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer1 = self.conv_init(buffer) + buffer

        # # Deep Spatial-Angular Correlation Learning
        # buffer = self.altblock(buffer,cdp) + buffer


        buffer = buffer1


        for block in self.altblock:
            buffer = block(buffer,cdp)

        
        buffer = buffer + buffer1

        # UP-Sampling
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=u, v=v)
        y = self.upsampling(buffer)
        y = rearrange(y, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)

        y = rearrange(y, 'b c u v h w -> b u v c h w')


        return y
    

#################################################################################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)






class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
from typing import Callable, Tuple, TypeVar

__all__ = ['DConv', 'SpatialConv', 'AngularConv']


T = TypeVar('T')
_spatial = Tuple[T, T]

class DConv(nn.Module):
    available_dims = ['u', 'v', 'h', 'w']
    all_dims = ['b', 'c'] + available_dims

    def __init__(self,
                 in_channels: int, out_channels: int, connection: str,
                 kernel_size: _spatial[int] = (3, 3),
                 stride: _spatial[int] = (1, 1),
                 padding: _spatial[int] = (1, 1),
                 n: int = 1,
                 act: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2, inplace=True),
                 *args, **kwargs):
        super(DConv, self).__init__()
        assert all(dim in self.available_dims for dim in connection)
        self.connection = connection
        self.dims_preserve, self.dims_active = [], []
        for dim in self.available_dims:
            _dst = self.dims_active if dim in connection else self.dims_preserve
            _dst.append(dim)
        self.pattern0 = f"{' '.join(['b', 'c'] + self.available_dims)}"
        self.pattern1 = f"({' '.join(['b'] + self.dims_preserve)}) {' '.join(['c'] + self.dims_active)}"

        self.body = []
        for i in range(n):
            self.body.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          padding_mode='reflect',
                          *args, **kwargs))
            if act:
                self.body.append(act())
        self.body = nn.Sequential(*self.body)

    def forward(self, inp: torch.Tensor):
        szs_preserve = {dim: sz for dim, sz in zip(self.all_dims, inp.shape) if dim in self.dims_preserve}
        x = rearrange(inp, f'{self.pattern0} -> {self.pattern1}')
        x = self.body(x)
        x = rearrange(x, f'{self.pattern1} -> {self.pattern0}', **szs_preserve)

        return x


def SpatialConv(*args, **kwargs) -> DConv:
    return DConv(connection='hw', *args, **kwargs)

    
#############################################################################################

def lf2token(lf: torch.Tensor) -> torch.Tensor:
    token = rearrange(lf, 'b c d d3 d4 -> d (b d3 d4) c')
    return token


def token2lf(token: torch.Tensor, d1: int, d2: int, d3: int, d4: int) -> torch.Tensor:
    lf = rearrange(token, '(d3 d4) (b d1 d2) (c) -> b c d1 d2 d3 d4', d1=d1, d2=d2, d3=d3, d4=d4)
    return lf


class AltFilter(nn.Module):
    available_dims = ['u', 'v', 'h', 'w']
    all_dims = ['b', 'c'] + available_dims

    def __init__(self, in_channels: int, connection: str, kernel_sz: int = 3,angRes=5):
        super(AltFilter, self).__init__()
        assert all(dim in self.available_dims for dim in connection)
        self.angRes = angRes
        self.connection = connection
        self.kernel_sz = kernel_sz
        self.dims_preserve, self.dims_active = [], []
        for dim in self.available_dims:
            _dst = self.dims_active if dim in connection else self.dims_preserve
            _dst.append(dim)
        self.pattern0 = f"{' '.join(['b', 'c'] + self.available_dims)}"
        self.pattern1 = f"{' '.join(['b', 'c'] + self.dims_preserve + self.dims_active)}"

        # if "w" in self.dims_active or "h" in self.dims_active:
        self.linear_in = nn.Linear(in_channels, in_channels, bias=True)
        self.norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=8,
            bias=True
        )
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2, bias=True),
            nn.ReLU(True),
            nn.Linear(in_channels * 2, in_channels, bias=True),
        )
        self.linear_out = nn.Linear(in_channels, in_channels, bias=True)
        self.conv = SpatialConv(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                n=3, bias=True)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
        )

    def forward(self, inp: torch.Tensor, cdp):
        # print(inp.shape)
        # print(self.pattern0, self.pattern1)
        inp1 = rearrange(inp, 'b c (u v) h w -> b c u v h w',u=self.angRes, v=self.angRes)
        szs_preserve = {dim: sz for dim, sz in zip(self.all_dims, inp.shape) if dim in self.dims_preserve}
        x1 = rearrange(inp1, f'{self.pattern0} -> {self.pattern1}')
        _, c, d1, d2, d3, d4 = x1.shape


        ##### 退化信息编码########
        cdp = self.fc(cdp) #((buv) c)
        cdp_spa = cdp.view(cdp.size(0), cdp.size(1), 1, 1) #((buv) c 1 1)
        gamma_spa, beta_spa = torch.chunk(cdp_spa, chunks=2, dim=1)

        buv,c1,h1,w1 = cdp_spa.shape
        cdp_ang = cdp_spa.expand(buv, c1, d1, d2)
        cdp_ang = rearrange(cdp_ang, '(b u v) c h w -> (b h w) c u v', u=self.angRes, v=self.angRes, h=d1, w=d2)
        gamma_ang, beta_ang = torch.chunk(cdp_ang, chunks=2, dim=1)
        

        # query
        x_q = rearrange(x1, 'b c d1 d2 d3 d4 -> b c (d3 d4) d1 d2', d1=d1, d2=d2)
        token_q = lf2token(x_q)
        token_q = self.linear_in(token_q)
        token_q_norm = self.norm(token_q)
        
		# key & value
        x = rearrange(x1, 'b c d1 d2 d3 d4 -> (b d1 d2) c d3 d4')
        x = (1 + gamma_ang) * x + beta_ang
        x_k = rearrange(x, '(b d1 d2) c d3 d4 -> b c (d3 d4) d1 d2', d1=d1, d2=d2)
        token_k = lf2token(x_k)
        token_k = self.linear_in(token_k)
        # token_k_norm = self.norm(token_k)
        


        # print('token_norm.shape is {}'.format(token_norm.shape))
        token = self.attention(query=token_q_norm,
                               key=token_q_norm,
                               value=token_k,
                               need_weights=False)[0] + token_q
        token = self.feed_forward(token) + token
        token = self.linear_out(token)

        x2 = token2lf(token, d1=d1, d2=d2, d3=d3, d4=d4)+x1
        

        x = rearrange(x2, 'b c d1 d2 d3 d4 -> (b d3 d4) c d1 d2')
        x = (1 + gamma_spa) * x + beta_spa
        x = rearrange(x, '(b d3 d4) c d1 d2 -> b c d1 d2 d3 d4', d3=d3, d4=d4)

        x = rearrange(x, f'{self.pattern1} -> {self.pattern0}', **szs_preserve)

        x2 = rearrange(x2, 'b c d1 d2 d3 d4 -> b c d3 d4 d1 d2')

        # print(x.shape)
        # print(x2.shape)

        x = self.conv(x) + x2

        # print(x.shape)
        x = rearrange(x, 'b c d1 d2 d3 d4 -> b c (d1 d2) d3 d4')
        
        return x 

####################################################################################################

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out, HR)

        return loss


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def weights_init(m):
    pass