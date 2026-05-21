import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from thop import profile


class FMC(nn.Module):

    def __init__(self, height, width, in_channels, out_channels, kernel_size=3, n_groups=8):
        super(FMC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_groups = n_groups

        # 创建频率到卷积核的映射层
        # 每个通道映射到k*k*d的维度
        self.frequency_mapper_m = nn.Conv1d(height * width, kernel_size * kernel_size * out_channels, kernel_size=1)
        self.frequency_mapper_p = nn.Conv1d(height * width, kernel_size * kernel_size * out_channels, kernel_size=1)

        # 创建分组权重
        self.group_weights = nn.Parameter(torch.ones(n_groups) / n_groups)

        # 创建用于调整输出的1x1卷积
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模块权重"""
        # 初始化频率映射层
        nn.init.kaiming_normal_(self.frequency_mapper_m.weight)
        if self.frequency_mapper_m.bias is not None:
            nn.init.zeros_(self.frequency_mapper_m.bias)

        nn.init.kaiming_normal_(self.frequency_mapper_p.weight)
        if self.frequency_mapper_p.bias is not None:
            nn.init.zeros_(self.frequency_mapper_p.bias)

            # 初始化输出卷积层
        nn.init.kaiming_normal_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

            # 初始化分组权重
        nn.init.constant_(self.group_weights, 1.0 / self.n_groups)

    def compute_fft(self, x):
        """对特征图 x 进行傅里叶变换，返回幅值和相位"""
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        m = torch.abs(x_fft)
        p = torch.angle(x_fft)
        return m, p

    def reshape_m_p(self, m, p):
        """
        将幅值和相位映射为卷积核权重
        """
        b, c, h, w = m.shape
        k = self.kernel_size
        d = self.out_channels

        # 将特征图重塑为[b, c, h*w]
        m_flat = m.reshape(b, h * w, c)
        p_flat = p.reshape(b, h * w, c)

        # 使用Conv1d进行特征变换
        m_transformed = self.frequency_mapper_m(m_flat)  # [b, k*k*d, c]
        p_transformed = self.frequency_mapper_p(p_flat)  # [b, k*k*d, c]

        # 重塑为[b, c*k, d*k]
        m_reshaped = m_transformed.reshape(b, c * k, d * k)
        p_reshaped = p_transformed.reshape(b, c * k, d * k)

        return m_reshaped, p_reshaped

    def group_frequencies(self, m):
        """
        将幅值 m 按照矩形环的方式进行分组
        m的形状为 [b, c*k, d*k]
        返回n_groups个分组后的幅值特征图，每个形状仍为 [b, c*k, d*k]
        """
        b, c_k, d_k = m.shape
        grouped_m = []

        # 创建n_groups个不同的矩形环掩码
        for i in range(self.n_groups):
            # 计算矩形环的边界
            outer_ratio = 1.0 - (i / self.n_groups)
            inner_ratio = 1.0 - ((i + 1) / self.n_groups)

            if i == self.n_groups - 1:  # 最内层包含中心点
                inner_ratio = 0.0

            # 中心点
            center_c = c_k // 2
            center_d = d_k // 2

            # 创建矩形环掩码 (使用网格索引而不是循环，加速计算)
            c_indices = torch.arange(c_k, device=m.device).view(1, -1, 1).repeat(b, 1, d_k)
            d_indices = torch.arange(d_k, device=m.device).view(1, 1, -1).repeat(b, c_k, 1)

            # 计算到中心的距离
            dist_c = torch.abs(c_indices - center_c) / (c_k / 2)
            dist_d = torch.abs(d_indices - center_d) / (d_k / 2)
            max_dist = torch.maximum(dist_c, dist_d)

            # 如果在当前矩形环内，则设置掩码为1
            mask = ((max_dist >= inner_ratio) & (max_dist <= outer_ratio)).float()

            # 应用掩码
            grouped_m.append(m * mask)

        return grouped_m

    def ifft_modulation(self, grouped_m, p):
        """
        使用相位 p 对每个分组的幅值进行逆傅里叶变换
        grouped_m: 形状为 [b, c*k, d*k] 的幅值分组列表
        p: 形状为 [b, c*k, d*k] 的相位图
        返回: 每组幅值对应的调制特征图，形状为 [d, c, k, k]
        """
        b = grouped_m[0].shape[0]
        c = self.in_channels
        k = self.kernel_size
        d = self.out_channels
        modulated_kernels = []

        for m_group in grouped_m:
            # 使用相位调制幅值
            complex_fft = m_group * torch.exp(1j * p)

            # 进行逆傅里叶变换
            modulated = torch.fft.ifft2(complex_fft, dim=(-2, -1)).real

            # 重塑为卷积核格式 [b, d, c, k, k]
            modulated = modulated.view(b, c, k, d, k)
            modulated = modulated.permute(0, 3, 1, 2, 4).contiguous()
            modulated = modulated.view(b, d, c, k, k)

            # 如果有批次维度，则取平均
            if b > 1:
                modulated = modulated.mean(dim=0)
            else:
                modulated = modulated.squeeze(0)

            modulated = torch.sigmoid(modulated)
            modulated_kernels.append(modulated)

        return modulated_kernels

    def generate_final_kernel(self, modulated_kernels):
        """
        对n个调制特征图加权求和，生成最终卷积核
        modulated_kernels: 形状为 [d, c, k, k] 的卷积核列表
        """
        # 使用softmax确保权重和为1
        weights = F.softmax(self.group_weights, dim=0)

        # 初始化一个与卷积核相同形状的张量
        final_kernel = torch.zeros_like(modulated_kernels[0])

        # 加权求和
        for i, kernel in enumerate(modulated_kernels):
            final_kernel += weights[i] * kernel

        return final_kernel

    def forward(self, x):
        """
        前向传播函数
        x: 输入特征图，形状为 [b, c, h, w]
        """
        # 保存输入尺寸
        b, c, h, w = x.shape

        # 1. 进行傅里叶变换
        m, p = self.compute_fft(x)

        # 2. 将幅值和相位reshape
        m_reshaped, p_reshaped = self.reshape_m_p(m, p)

        # 3. 对幅值进行频率分组
        grouped_m = self.group_frequencies(m_reshaped)

        # 4. 对每组幅值进行逆傅里叶变换
        modulated_kernels = self.ifft_modulation(grouped_m, p_reshaped)

        # 5. 加权求和生成最终卷积核
        final_kernel = self.generate_final_kernel(modulated_kernels)

        # 6. 与输入特征图进行卷积
        output = F.conv2d(x, final_kernel, padding=self.kernel_size // 2)

        # 使用1x1卷积调整输出通道
        output = self.out_conv(output)

        # 返回与输入特征图相同尺寸的输出
        return output

if __name__ == '__main__':
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    in_channels = 4
    height, width = 110, 110
    out_channels = 4
    kernel_size = 3
    n_groups = 4

    # 创建输入数据
    input_data = torch.randn(1, in_channels, height, width).to(device)

    # 创建模型并移动到相同设备
    model = FMC(
        width=width,
        height=height,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        n_groups=n_groups
    ).to(device)

    # 计算FLOPs和参数数量
    flops, params = profile(model, inputs=(input_data,))
    print('params:', params)
    print('flops:', flops)


