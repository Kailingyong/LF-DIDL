import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange




# 将 MultiDegrade 继承自 nn.Module
class MultiDegrade(nn.Module):
    def __init__(self, scale=4, kernel_size=21, sig_min=0, sig_max=4.0, noise=None, sig=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.scale = scale

        # 确保 sig_min, sig_max 的赋值逻辑不变
        if sig is not None:
            self.sig_min, self.sig_max = sig, sig
        else:
            self.sig_min, self.sig_max = sig_min, sig_max

        # 将 gen_kernel, blur_by_kernel, bicubic 作为 nn.Module 的子模块
        # 这样在 MultiDegrade.to(device) 时，它们也会被移动到正确的设备
        self.gen_kernel = Isotropic_Gaussian_Blur(kernel_size, [self.sig_min, self.sig_max])
        self.blur_by_kernel = Blur_by_Kernel(kernel_size=kernel_size)
        # self.bicubic_interp = Bicubic() # 避免和 forward 中的 bicubic 方法名冲突

        if noise is not None:
            self.noise, self.noise_rand = noise, 0  # inference stage
        else:
            self.noise, self.noise_rand = 75, 1   # training stage

    @torch.no_grad() # 降质过程通常不需要梯度
    def forward(self, hr):
        # 获取输入 Tensor 的设备，确保后续新创建的 Tensor 都在同一设备
        B, N, C, H, W = hr.size()
        device = hr.device

        # --- 1. 生成 kernel ---
        # gen_kernel 已经被初始化为 nn.Module 的子模块，其内部的 xx_yy 也是 buffer
        # 当 MultiDegrade 被移动到 GPU 时，gen_kernel 及其 buffer 也会在 GPU 上。
        # 确保 `batch` 参数 (hr.size(0)) 的值被传递，以便生成 B 个核
        kernels, sigma = self.gen_kernel(B, device) # 传递 device
        
        # 原始代码中这里有 kernels, sigma = kernels.to(hr.device), sigma.to(hr.device)
        # 优化后，gen_kernel 内部就已经在目标设备上生成了，所以这行可以省略
        # 但为了绝对安全和与原始逻辑的对齐，如果 gen_kernel 内部没有完全保证，留着也无妨。
        # 在这里的优化版 Isotropic_Gaussian_Blur 中，已经确保了在 device 上生成。

        hr = rearrange(hr, 'b n c h w -> b (n c) h w')

        # --- 2. 模糊 ---
        # blur_by_kernel 已经被初始化为 nn.Module 的子模块，会在 GPU 上运行
        hr_blured = self.blur_by_kernel(hr, kernels)


        # # --- 3. 下采样 ---
        # # bicubic_interp 已经被初始化为 nn.Module 的子模块，会在 GPU 上运行
        # # 确保其内部创建的 Tensor 也在 GPU 上
        # lr_blured = self.bicubic_interp(hr_blured, scale=1/self.scale, device=device) # 传递 device

        # === 3. 下采样：直接换成 CUDA 优化版本 ===
        lr_blured = F.interpolate(hr_blured, scale_factor=1.0 / self.scale, mode='bicubic', align_corners=False)

        lr_blured = rearrange(lr_blured, 'b (n c) h w -> b n c h w', n=N, c=C)

        # --- 4. 噪声 ---
        if self.noise > 0:
            # 确保 noise_level 直接在目标设备上创建
            if self.noise_rand == 1:
                noise_level = torch.rand(B, 1, 1, 1, 1, device=device) * (self.noise / 75)
            else:  ## 测试阶段
                noise_level = torch.ones(B, 1, 1, 1, 1, device=device) * (self.noise / 75)
            # torch.randn_like 会自动在 lr_blured 的设备上生成噪声
            noise = torch.randn_like(lr_blured).mul_(noise_level * 75 / 255)
            lr_blured.add_(noise)
        else:
            # 确保 noise_level 直接在目标设备上创建
            noise_level = torch.zeros(B, 1, 1, 1, 1, device=device)

        return lr_blured, sigma, noise_level.squeeze(1)


# Blur_by_Kernel 已经是 nn.Module，保持不变
class Blur_by_Kernel(nn.Module):
    def __init__(self, kernel_size=21):
        super(Blur_by_Kernel, self).__init__()
        self.kernel_size = kernel_size
        if kernel_size % 2 == 1:
            self.padding = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.padding = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))

    def forward(self, input, kernel):
        [B, C, H, W] = input.size()
        input_pad = self.padding(input)
        # 这里的卷积操作会保留其原始行为（可能不是普遍期望的）
        output = F.conv2d(input_pad.transpose(0, 1), kernel.unsqueeze(1), groups=B)
        output = output.transpose(0, 1)
        return output


# Isotropic_Gaussian_Blur 也改为 nn.Module，并注册 xx_yy 为 buffer
class Isotropic_Gaussian_Blur(nn.Module):
    def __init__(self, kernel_size, sig_range):
        super(Isotropic_Gaussian_Blur, self).__init__()
        self.kernel_size = kernel_size
        self.sig_min = sig_range[0]
        self.sig_max = sig_range[1]
        
        # 注册 xx_yy 为 buffer，确保它在 GPU 上

        ax = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        xx = ax.repeat(self.kernel_size).view(1, self.kernel_size, self.kernel_size)
        yy = ax.repeat_interleave(self.kernel_size).view(1, self.kernel_size, self.kernel_size)



        self.register_buffer('xx_yy', -(xx ** 2 + yy ** 2)) # (k, k)

    # 接收 device 参数，确保 sigma 在指定设备上生成
    def forward(self, batch_size, device): # 修改 __call__ 为 forward，并接收 device
        sig_min, sig_max = self.sig_min, self.sig_max
        # 确保 sigma 在目标设备上生成
        sigma = torch.rand(batch_size, device=device) * (sig_max - sig_min) + sig_min + 1e-6

        # xx_yy 已经是 buffer，会在 GPU 上
        kernel = torch.exp(self.xx_yy / (2. * sigma.view(-1, 1, 1) ** 2))
        kernel = kernel / kernel.sum([1, 2], keepdim=True)
        return kernel, sigma




if __name__ == "__main__":
    gen_LR = MultiDegrade()
    angRes = 5
    input = torch.randn(4, 25, 3, 128, 128).cuda()
    output = gen_LR(input)
