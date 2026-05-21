import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment


# SAD loss
class SADLoss(nn.Module):
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        # 确保输入形状正确 (Batch, Bands, Pixels)
        if len(y_pred.shape) > 2:
            # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
            if len(y_pred.shape) == 4:
                b, c, h, w = y_pred.shape
                y_true = y_true.view(b, c, -1)
                y_pred = y_pred.view(b, c, -1)
            # Or if input is already (B, C, N)

        # 为了计算方便，转置为 (Batch, Pixels, Bands)
        # 这里的 view(-1, ...) 假设 batch_size=1 或者合并 batch 计算
        # 安全起见，我们按 (N_pixels, 1, Bands) 处理，利用 bmm
        y_true = y_true.transpose(1, 2).reshape(-1, 1, y_true.shape[1])  # (N, 1, Bands)
        y_pred = y_pred.transpose(1, 2).reshape(-1, 1, y_pred.shape[1])  # (N, 1, Bands)

        # 1. 加上 epsilon 防止除以 0
        epsilon = 1e-7
        y_true_norm = torch.sqrt(torch.bmm(y_true, y_true.permute(0, 2, 1))) + epsilon
        y_pred_norm = torch.sqrt(torch.bmm(y_pred, y_pred.permute(0, 2, 1))) + epsilon

        summation = torch.bmm(y_pred, y_true.permute(0, 2, 1))

        # 2. 计算余弦值
        cos_val = summation / (y_true_norm * y_pred_norm)

        # 3. 【关键修复】截断数值，防止越界导致 NaN
        cos_val = torch.clamp(cos_val, -1.0 + epsilon, 1.0 - epsilon)

        angle = torch.acos(cos_val)
        sad = torch.mean(angle)
        return sad


class FrequencyDomainLoss(nn.Module):
    """
    频域损失函数：包含幅度损失和相位损失
    先对光谱维度降维，再进行FFT变换
    """

    def __init__(self,
                 num_bands,
                 reduced_dim=32,
                 amp_threshold_percentile=75,
                 epsilon=1e-8,
                 weight_mag=1.0,
                 weight_phase=1.0):
        """
        Args:
            num_bands: 原始光谱波段数
            reduced_dim: 降维后的维度
            amp_threshold_percentile: 振幅掩码的百分位阈值(0-100)
            epsilon: 数值稳定性常数
            weight_mag: 幅度损失权重
            weight_phase: 相位损失权重
        """
        super(FrequencyDomainLoss, self).__init__()
        self.num_bands = num_bands
        self.reduced_dim = reduced_dim
        self.amp_threshold_percentile = amp_threshold_percentile
        self.epsilon = epsilon
        self.weight_mag = weight_mag
        self.weight_phase = weight_phase

        # 生成随机正交投影矩阵并注册为buffer（不参与训练）
        projection = self._generate_random_projection()
        self.register_buffer('projection_matrix', projection)

    def _generate_random_projection(self):
        """
        生成随机正交投影矩阵用于光谱降维
        使用QR分解保证正交性
        """
        # 生成随机矩阵
        random_matrix = torch.randn(self.num_bands, self.reduced_dim)
        # QR分解得到正交矩阵
        Q, _ = torch.linalg.qr(random_matrix)
        return Q  # shape: (num_bands, reduced_dim)

    def spectral_reduction(self, x):
        """
        对光谱维度进行降维
        Args:
            x: shape (B, C, H, W) 或 (B, H*W, C)
        Returns:
            reduced: shape (B, reduced_dim, H, W) 或 (B, H*W, reduced_dim)
        """
        original_shape = x.shape
        device = x.device  # 获取输入数据的设备

        # 确保投影矩阵在正确的设备上
        projection_matrix = self.projection_matrix.to(device)

        if len(x.shape) == 4:  # (B, C, H, W)
            B, C, H, W = x.shape
            # Reshape to (B*H*W, C)
            x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
            # Apply projection: (B*H*W, C) @ (C, reduced_dim) -> (B*H*W, reduced_dim)
            x_reduced = torch.matmul(x_reshaped, projection_matrix)
            # Reshape back to (B, H, W, reduced_dim) -> (B, reduced_dim, H, W)
            x_reduced = x_reduced.reshape(B, H, W, self.reduced_dim).permute(0, 3, 1, 2)

        elif len(x.shape) == 3:  # (B, N, C)
            B, N, C = x.shape
            # Reshape to (B*N, C)
            x_reshaped = x.reshape(-1, C)
            # Apply projection
            x_reduced = torch.matmul(x_reshaped, projection_matrix)
            # Reshape back to (B, N, reduced_dim)
            x_reduced = x_reduced.reshape(B, N, self.reduced_dim)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        return x_reduced

    def spatial_fft(self, x):
        """
        对空间维度进行1D FFT变换
        Args:
            x: shape (B, C, H, W)
        Returns:
            fft_result: complex tensor, shape (B, C, H, W//2+1)
        """
        # 对宽度维度进行实数FFT
        fft_result = torch.fft.rfft(x, dim=-1)
        return fft_result

    def compute_amplitude_mask(self, magnitude):
        """
        计算振幅掩码：只保留高于阈值的频率分量
        Args:
            magnitude: shape (B, C, H, W_freq)
        Returns:
            mask: binary mask, shape (B, C, H, W_freq)
        """
        # 展平magnitude进行百分位计算
        mag_flat = magnitude.reshape(-1)

        # 计算阈值（使用百分位数）
        # 使用torch.quantile，确保在正确的设备上
        threshold = torch.quantile(mag_flat,
                                   self.amp_threshold_percentile / 100.0)

        # 创建二值掩码
        mask = (magnitude > threshold).float()

        return mask

    def forward(self, y_true, y_pred):
        """
        计算频域损失
        Args:
            y_true: 真值图像, shape (B, C, H, W) 或 (B, N, C)
            y_pred: 预测图像, shape (B, C, H, W) 或 (B, N, C)
        Returns:
            loss_freq: 总频域损失
            loss_mag: 幅度损失
            loss_phase: 相位损失
        """
        device = y_pred.device

        # 1. 光谱降维
        y_true_reduced = self.spectral_reduction(y_true)
        y_pred_reduced = self.spectral_reduction(y_pred)

        # 确保是4D张量 (B, C, H, W)
        if len(y_true_reduced.shape) == 3:
            B, N, C = y_true_reduced.shape
            H = W = int(np.sqrt(N))
            y_true_reduced = y_true_reduced.permute(0, 2, 1).reshape(B, C, H, W)
            y_pred_reduced = y_pred_reduced.permute(0, 2, 1).reshape(B, C, H, W)

        # 2. 对空间维度进行FFT
        Y_fft = self.spatial_fft(y_true_reduced)  # Complex tensor
        Y_hat_fft = self.spatial_fft(y_pred_reduced)  # Complex tensor

        # 3. 计算幅度
        Y_mag = torch.abs(Y_fft)  # |Y'|
        Y_hat_mag = torch.abs(Y_hat_fft)  # |Ŷ'|

        # 4. 计算幅度损失
        loss_mag = F.mse_loss(Y_mag, Y_hat_mag)

        # 5. 计算单位复数向量（用于相位对齐）
        U = Y_fft / (Y_mag + self.epsilon)  # 单位相量
        U_hat = Y_hat_fft / (Y_hat_mag + self.epsilon)

        # 6. 计算振幅掩码
        M_amp = self.compute_amplitude_mask(Y_mag)

        # 7. 计算相位损失（使用余弦相似度）
        # 复数内积的实部 = cos(相位差)
        cos_similarity = torch.real(U * torch.conj(U_hat))

        # 应用掩码并计算加权平均
        masked_similarity = cos_similarity * M_amp

        # 添加epsilon避免除零
        mask_sum = M_amp.sum()
        if mask_sum > 0:
            loss_phase = 1 - (masked_similarity.sum() / (mask_sum + self.epsilon))
        else:
            # 如果掩码全为0，返回0损失
            loss_phase = torch.tensor(0.0, device=device)

        # 8. 组合损失
        loss_freq = self.weight_mag * loss_mag + self.weight_phase * loss_phase

        return loss_freq, loss_mag, loss_phase


class My_Loss(nn.Module):
    def __init__(self, num_bands=188, weight_mse=0, weight_sad=2, weight_endm=0.001,
                 weight_aban=1e-3, weight_fft=1e-7, weight_amp=1, weight_phase=1):
        super(My_Loss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_sad = weight_sad
        self.weight_endm = weight_endm
        self.weight_aban = weight_aban
        self.weight_fft = weight_fft
        self.SAD = SADLoss()
        self.MSE = nn.MSELoss()

        # 初始化频域损失模块
        if weight_fft > 0:
            self.freq_loss = FrequencyDomainLoss(
                num_bands=num_bands,
                reduced_dim=min(5, num_bands // 2),  # 自适应降维维度
                amp_threshold_percentile=5,
                weight_mag=weight_amp,
                weight_phase=weight_phase
            )
        else:
            self.freq_loss = None

    def forward(self, y_true, y_pred, endm=None, hsi_mean=None, pred_aban=None):
        device = y_pred.device
        loss = torch.tensor(0.0, device=device)
        loss_sad = torch.tensor(0.0, device=device)
        loss_mse = torch.tensor(0.0, device=device)
        loss_endm = torch.tensor(0.0, device=device)
        loss_aban = torch.tensor(0.0, device=device)
        loss_fft = torch.tensor(0.0, device=device)

        if self.weight_mse != 0:
            loss_mse = self.weight_mse * self.MSE(y_true, y_pred)
            loss += loss_mse

        # 调整数据形状
        if 1 < len(y_pred.shape) < 5:
            y_true_shaped = y_true.view(y_true.shape[0], y_true.shape[1], -1).transpose(1, 2)
            y_pred_shaped = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1).transpose(1, 2)
        elif len(y_pred.shape) >= 5:
            from einops import rearrange
            y_true_shaped = rearrange(y_true, 'n b c w h -> (n b) (w h) c')
            y_pred_shaped = rearrange(y_pred, 'n b c w h -> (n b) (w h) c')
        else:
            y_true_shaped = y_true
            y_pred_shaped = y_pred

        if self.weight_sad != 0:
            loss_sad = self.weight_sad * self.SAD(y_true_shaped, y_pred_shaped)
            loss += loss_sad

        # 使用新的频域损失
        if self.weight_fft != 0 and self.freq_loss is not None:
            try:
                loss_freq_total, loss_mag, loss_phase = self.freq_loss(y_true, y_pred)
                loss_fft = self.weight_fft * loss_freq_total
                loss += loss_fft
            except Exception as e:
                print(f"Frequency loss computation failed: {e}")
                loss_fft = torch.tensor(0.0, device=device)

        if endm is not None and hsi_mean is not None and self.weight_endm != 0:
            loss_endm = self.weight_endm * self.MSE(hsi_mean, endm)
            loss += loss_endm

        if pred_aban is not None and self.weight_aban != 0:
            aban_norm = torch.norm(pred_aban, p=0.5, dim=1)
            loss_aban = self.weight_aban * aban_norm.mean()
            loss += loss_aban

        # 检查NaN/Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Warning: Final loss contains NaN or Inf values.")
            return (torch.tensor(0.0, device=device, requires_grad=True),) * 5 + \
                (torch.tensor(1.0, device=device, requires_grad=True),)

        return loss_sad, loss_mse, loss_endm, loss_aban, loss_fft, loss



def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c
    for i in range(c):
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]) ** 2).sum() / (w * h))
    mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    return class_rmse, mean_rmse

# 会遇到预测的abu顺序与gt不对应的情况
def compute_rmse_with_best_matching(x_true, x_pre):
    """
    使用匈牙利算法找到最佳丰度图匹配
    Args:
        x_true: 真值丰度 (H, W, num_endm)
        x_pre: 预测丰度 (H, W, num_endm)
    Returns:
        class_rmse: 每个端元的RMSE列表
        mean_rmse: 平均RMSE
        best_order: 最佳匹配顺序
    """
    w, h, c = x_true.shape

    # 构建代价矩阵：计算所有丰度图对之间的RMSE
    cost_matrix = np.zeros((c, c))

    for i in range(c):
        for j in range(c):
            cost_matrix[i, j] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, j]) ** 2).sum() / (w * h))

    # 使用匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # col_ind就是最佳匹配顺序
    best_order = col_ind.tolist()

    # 计算最佳匹配下的RMSE
    class_rmse = [0] * c
    for i in range(c):
        matched_idx = best_order[i]
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, matched_idx]) ** 2).sum() / (w * h))

    mean_rmse = np.sqrt(sum([r ** 2 for r in class_rmse]) / c)

    return class_rmse, mean_rmse, best_order


def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad

# 顺序对不上


def compute_sad_with_best_matching(inp, target):
    """
    计算预测端元与真实端元之间的最佳匹配 SAD (Spectral Angle Distance)。

    Args:
        inp: 预测端元矩阵 (Predicted Endmembers), shape: (n_bands, n_endmembers)
        target: 真实端元矩阵 (Ground Truth), shape: (n_bands, n_endmembers)

    Returns:
        sad_err: 排序对齐后的每个端元的 SAD 值列表
        mean_sad: 平均 SAD 值
        best_order: 预测端元对应的最佳索引顺序 (用于后续对齐绘图)
    """

    # 1. 【新增】强制清洗数据，防止 NaN 传入导致 linear_sum_assignment 崩溃
    if np.isnan(inp).any():
        # print("Warning: NaNs detected in predicted endmembers. Replacing with 0.")
        inp = np.nan_to_num(inp)

    if np.isnan(target).any():
        # print("Warning: NaNs detected in target endmembers. Replacing with 0.")
        target = np.nan_to_num(target)

    # 获取维度：b=波段数, p=端元数
    b, p = inp.shape

    # 2. 构建代价矩阵 (Cost Matrix)
    # 矩阵大小为 (p, p)，其中 cost_matrix[i, j] 表示 "真实端元 i" 与 "预测端元 j" 之间的 SAD
    cost_matrix = np.zeros((p, p))

    for i in range(p):  # 遍历真实端元 (Target)
        for j in range(p):  # 遍历预测端元 (Input)
            # 获取向量
            vec_target = target[:, i]
            vec_inp = inp[:, j]

            # 计算范数 (L2 Norm)
            norm_target = np.linalg.norm(vec_target)
            norm_inp = np.linalg.norm(vec_inp)

            # 计算点积
            summation = np.dot(vec_target, vec_inp)

            # 计算余弦值 (加上 1e-9 防止分母为 0)
            cos_val = summation / (norm_target * norm_inp + 1e-9)

            # 【关键】截断数值范围到 [-1, 1]
            # 浮点数误差可能导致 cos_val 变成 1.0000001，直接 acos 会产生 NaN
            cos_val = np.clip(cos_val, -1.0, 1.0)

            # 计算 SAD (反余弦)
            cost_matrix[i, j] = np.arccos(cos_val)

    # 3. 使用匈牙利算法进行最佳匹配
    # linear_sum_assignment 寻找让总 SAD 最小的行列组合
    # row_ind 对应真实端元索引 (0, 1, 2...), col_ind 对应匹配到的预测端元索引
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # best_order 即为最佳匹配的预测端元索引列表
    best_order = col_ind.tolist()

    # 4. 提取对齐后的结果
    sad_err = []
    for i in range(p):
        # 取出最佳匹配位置的 SAD 值
        # cost_matrix[真实索引 i, 匹配到的预测索引]
        sad = cost_matrix[i, best_order[i]]
        sad_err.append(sad)

    mean_sad = np.mean(sad_err)

    return sad_err, mean_sad, best_order


