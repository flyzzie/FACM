from thop import profile
from torch import nn
from mamba_ssm import Mamba
from .trial.FANLayer import FANLayer
from .trial.FMC import FMC

class SumToOne(nn.Module):
    def __init__(self, scale=3.5):
        super(SumToOne, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = torch.softmax(self.scale * x, dim=1)
        return x

class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.chunk_size=None
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=channels,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=8,  # Block expansion factor
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                # nn.Linear(in_features=channels, out_features=channels),
                FANLayer(channels, channels),
                nn.LayerNorm(channels),
                nn.SiLU()
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape

        # 计算需要的填充
        flat_len = B * H * W
        if self.chunk_size:
            remainder = flat_len % self.config.chunk_size
            if remainder != 0:
                # 需要填充
                pad_len = self.config.chunk_size - remainder
                # 重塑和填充
                x_flat = x_re.view(-1, C)
                padding = torch.zeros(pad_len, C, device=x.device)
                x_flat = torch.cat([x_flat, padding], dim=0)
                x_flat = x_flat.unsqueeze(0)  # 添加批次维度
            else:
                x_flat = x_re.view(1, -1, C)
        else:
            # 不需要填充
            x_flat = x_re.view(1, -1, C)

        # 前向传播
        # x_flat, _ = self.mamba(x_flat)
        x_flat = self.mamba(x_flat)

        if self.use_proj:
            x_flat = self.proj(x_flat)

        # 如果进行了填充，需要去除
        if self.chunk_size:
            x_flat = x_flat[0, :flat_len]

        # 重塑回原始形状
        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()

        if self.use_residual:
            return x_recon + x
        else:
            return x_recon

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveGaussianMask(nn.Module):
    def __init__(self, num_channels, initial_sigma_x=30.0, initial_sigma_y=30.0, initial_rotation=0.0,
                 learn_rotation=True):
        super(AdaptiveGaussianMask, self).__init__()

        # Create channel-specific parameters
        self.log_sigma_x = nn.Parameter(torch.ones(num_channels) * torch.tensor(np.log(np.exp(initial_sigma_x) - 1)))
        self.log_sigma_y = nn.Parameter(torch.ones(num_channels) * torch.tensor(np.log(np.exp(initial_sigma_y) - 1)))

        # 旋转参数也按通道分别配置
        if learn_rotation:
            self.rotation = nn.Parameter(torch.ones(num_channels) * torch.tensor(initial_rotation, dtype=torch.float32))
        else:
            self.register_buffer('rotation',
                                 torch.ones(num_channels) * torch.tensor(initial_rotation, dtype=torch.float32))

        self.num_channels = num_channels

    def get_sigmas(self):
        """获取实际的sigma值（确保为正）"""
        sigma_x = F.softplus(self.log_sigma_x)  # [num_channels]
        sigma_y = F.softplus(self.log_sigma_y)  # [num_channels]
        return sigma_x, sigma_y

    def create_gaussian_masks(self, shape):
        """为每个通道创建高斯掩码"""
        h, w = shape
        center_y, center_x = h // 2, w // 2

        # 获取sigma值
        sigma_x, sigma_y = self.get_sigmas()  # [num_channels]
        rotation = self.rotation  # [num_channels]

        # 创建网格坐标，确保 float32 类型
        y_indices = torch.arange(0, h, device=self.log_sigma_x.device, dtype=torch.float32)
        x_indices = torch.arange(0, w, device=self.log_sigma_x.device, dtype=torch.float32)
        y, x = torch.meshgrid(y_indices - center_y, x_indices - center_x, indexing='ij')

        # 为每个通道创建一个掩码
        masks = []
        for c in range(self.num_channels):
            # 应用旋转变换
            if rotation[c] != 0:
                x_rot = x * torch.cos(rotation[c]) + y * torch.sin(rotation[c])
                y_rot = -x * torch.sin(rotation[c]) + y * torch.cos(rotation[c])
                x_c, y_c = x_rot, y_rot
            else:
                x_c, y_c = x, y

                # 计算高斯函数
            mask = torch.exp(-(x_c ** 2 / (2 * sigma_x[c] ** 2) + y_c ** 2 / (2 * sigma_y[c] ** 2)))
            masks.append(mask)

            # 堆叠所有掩码 [num_channels, h, w]
        return torch.stack(masks, dim=0)

    def forward(self, x):
        """
        应用通道级掩码分离低频和高频

        参数:
        x: 输入特征图 [batch_size, channels, height, width]

        返回:
        low_freq: 低频部分
        high_freq: 高频部分
        masks: 使用的掩码
        """
        batch_size, channels, h, w = x.shape

        # 确保通道数匹配
        assert channels == self.num_channels, f"Input has {channels} channels but mask is configured for {self.num_channels} channels"

        # 创建通道级掩码 [num_channels, h, w]
        masks = self.create_gaussian_masks((h, w))

        # 扩展掩码维度以匹配输入 [1, num_channels, h, w]
        masks = masks.unsqueeze(0)

        # 应用掩码分离低频和高频部分
        low_freq = x * masks
        high_freq = x * (1 - masks)

        return low_freq, high_freq, masks


class FFTModule(nn.Module):
    def __init__(self):
        super(FFTModule, self).__init__()

    def forward(self, x):
        # 确保输入是 float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

            # FFT变换
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 分离幅值和相位
        magnitude = torch.abs(x_fft_shifted)
        phase = torch.angle(x_fft_shifted)

        # 使用对数幅值，避免数值问题
        log_magnitude = torch.log(magnitude + torch.tensor(1e-10, dtype=torch.float32))

        return log_magnitude, phase, x_fft_shifted


class IFFTModule(nn.Module):
    def __init__(self):
        super(IFFTModule, self).__init__()

    def forward(self, magnitude, phase, use_zero_phase=False):
        # 确保输入是 float32
        if magnitude.dtype != torch.float32:
            magnitude = magnitude.to(torch.float32)
        if phase.dtype != torch.float32:
            phase = phase.to(torch.float32)

            # 重建复数FFT
        if use_zero_phase:
            # 如果使用零相位，只保留幅值信息
            complex_fft = torch.exp(magnitude)
        else:
            # 否则结合幅值和相位
            complex_fft = torch.exp(magnitude) * torch.exp(1j * phase)

            # 反向FFT变换
        x_ifft_shifted = torch.fft.ifftshift(complex_fft, dim=(-2, -1))
        x_ifft = torch.fft.ifft2(x_ifft_shifted, dim=(-2, -1))

        # 取实部作为结果，确保为 float32
        return torch.real(x_ifft).to(torch.float32)


class FrequencyDomainMambaBlock(nn.Module):
    def __init__(self, channels, use_residual=True):
        super(FrequencyDomainMambaBlock, self).__init__()

        self.use_residual = use_residual
        self.channels = channels

        # FFT和IFFT模块
        self.fft_module = FFTModule()
        self.ifft_module = IFFTModule()

        self.freq_module = AdaptiveGaussianMask(num_channels=channels,initial_sigma_x=1.5, initial_sigma_y=1.5,initial_rotation=1)

        # 其余部分保持不变
        self.low_freq_mamba = nn.Sequential(
            SpaMamba(channels, use_residual=use_residual),
        )

        self.high_freq_mamba = nn.Sequential(
            SpaMamba(channels, use_residual=use_residual),
        )

        self.all_freq_mamba = nn.Sequential(
            SpaMamba(channels, use_residual=use_residual),
        )

        # 可学习的通道级加权系数
        self.freq_weights = nn.Parameter(torch.ones(channels, 3, 1, 1, dtype=torch.float32))

    def forward(self, x):
        # 保存输入，用于可能的残差连接
        identity = x

        # 1. FFT变换
        log_magnitude, phase, original_fft = self.fft_module(x)

        # 2. 应用掩码分离低频和高频
        low_freq, high_freq, mask = self.freq_module(log_magnitude)
        # low_freq, high_freq, mask = self.mask_module(x)

        # 3. 分别使用Mamba处理低频和高频
        low_freq_processed = self.low_freq_mamba(low_freq)
        high_freq_processed = self.high_freq_mamba(high_freq)
        all_freq_processed = self.all_freq_mamba(log_magnitude)

        weights = F.softmax(self.freq_weights, dim=1)  # [C, 3, 1, 1]
        low_weights = weights[:, 0:1]  # [C, 1, 1, 1]
        high_weights = weights[:, 1:2]  # [C, 1, 1, 1]
        all_weights = weights[:, 2:3]   # [C, 1, 1, 1]

        # 去除多余维度
        low_weights = low_weights.squeeze(1)  # [C, 1, 1]
        high_weights = high_weights.squeeze(1)  # [C, 1, 1]
        all_weights = all_weights.squeeze(1)  # [C, 1, 1]

        # 应用权重
        weighted_low_freq = low_freq_processed * low_weights
        weighted_high_freq = high_freq_processed * high_weights
        weighted_all_freq = all_freq_processed * all_weights

        # 合并加权的频率成分
        combined_freq = weighted_low_freq + weighted_high_freq + weighted_all_freq

        # 5. IFFT变换回空间域
        output = self.ifft_module(combined_freq, phase)

        # 6. 残差连接（如果启用）
        if self.use_residual:
            output = output + identity

        return output

class FACMBackbone(nn.Module):
    def __init__(self, height,width, in_channels=128, hidden_dim=64, out_dim=32, num_classes=10, use_residual=True,
                 group_num=4, use_att=True, scale=3.5, ds=4, dropout=5e-2, use_zero_phase=False):
        super(FACMBackbone, self).__init__()
        self.height = height
        self.width = width
        self.dropout = dropout
        self.use_zero_phase = use_zero_phase
        self.use_residual = use_residual

        self.patch_embedding_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        self.fmc = nn.Sequential(
            FMC(
                width=width,
                height=height,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                n_groups=4
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 替换标准Mamba块为频域Mamba块
        self.mamba_1 = FrequencyDomainMambaBlock(
            channels=hidden_dim,
            use_residual=use_residual,
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            SumToOne(scale=scale),
        )

        # self.cls_head = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(num_classes),
        #     nn.SiLU(),
        #     SumToOne(scale=scale),
        # )

        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 特征嵌入
        x = self.patch_embedding_1(x)

        y = self.fmc(x)

        # 频域Mamba处理
        x = self.mamba_1(x)

        x = self.relu(self.bn(x)+y)

        # 分类头
        abun_get = self.cls_head(x)

        # 训练期间添加Dropout
        if self.training:
            abun_get = F.dropout2d(abun_get, p=self.dropout)

        return abun_get


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # batch, length, dim = 2, 110 * 100, 285
    # x = torch.randn(batch, length, dim).to("cuda")
    # print(x.shape)
    # model = Mamba(
    #     # This module uses roughly 3 * expand * d_model^2 parameters
    #     d_model=dim,  # Model dimension d_model
    #     d_state=16,  # SSM state expansion factor
    #     d_conv=4,  # Local convolution width
    #     expand=2,  # Block expansion factor
    # ).to("cuda")
    # y = model(x)
    # print(y.shape)
    #
    # num_endmember = 3
    # num_band = 285
    # rows = [100, 100]
    # # rows = 100

    # print(model)
    # # summary(model, [num_band, 100, 100])
    # input_data = torch.randn(1, num_band, rows[0], rows[1]).to(device)
    # pred_abun = model(input_data)
    # print(pred_abun.shape)
    input_data = torch.randn(1, 285, 110, 110).to(device)
    model = FACMBackbone(in_channels=285, num_classes=4).to(device)
    flops, params = profile(model, inputs=(input_data,))
    print('params:', params)
    print('flops:', flops)