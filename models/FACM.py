import torch
import torch.nn.functional as F
from thop import profile
from torch import nn
from .FACM_Backbone import FACMBackbone
from einops import rearrange
from torchsummary import summary
import math

import pywt
import numpy as np

class FACM(nn.Module):
    def __init__(self,  height, width, num_band, d_model=16, num_endm=3, num_queries_times=30, scale=3.5, ds=4, dropout=5e-2,device=None):
        super(FACM, self).__init__()
        self.height = height
        self.width = width
        self.num_band = num_band
        self.num_endm = num_endm
        self.num_queries_per_endm=num_queries_times
        self.down_ratio = ds
        self.backbone = FACMBackbone(
            self.height, self.width,
            in_channels=num_band, hidden_dim=d_model, num_classes=num_endm,
                                        scale=scale, ds=ds, dropout=dropout)

        self.num_queries = num_queries_times * num_endm
        self.query_embed = nn.Embedding(self.num_queries, num_band)
        self.weights = nn.Parameter(torch.ones((num_endm, num_queries_times)))
        self._reset_parameters()
        self.gate = nn.Parameter(torch.ones(self.num_endm, 1))
        self.kernel_temp = nn.Parameter(torch.ones(self.num_endm) * 1.0)
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        _, _, w, h = x.shape

        abun_get = self.backbone(x)
        endm_get = self.get_endmember_frequency(x)
        recon_linear = torch.einsum('brhw,rl->blhw', [abun_get, endm_get])
        return recon_linear + 1e-7, abun_get, endm_get


    def get_endmember_frequency(self, feature_map):
        """
        基于频率域分析和空间切块的端元生成器

        参数:
        - feature_map: 形状为[B, C, H, W]的特征图，其中C是频谱波段数

        返回:
        - endmembers: 形状为[num_endm, C]的端元矩阵
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # 获取参数
        num_endm = self.num_endm  # 端元数量
        num_queries_per_endm = self.num_queries_per_endm  # 每个端元的频率特征数

        # 步骤1: 空间切块处理
        # 将图像分成num_blocks×num_blocks个块
        num_blocks = 4
        block_h = H // num_blocks
        block_w = W // num_blocks

        # 收集所有块的特征
        block_features = []
        for i in range(num_blocks):
            for j in range(num_blocks):
                # 提取块
                h_start, h_end = i * block_h, (i + 1) * block_h
                w_start, w_end = j * block_w, (j + 1) * block_w
                block = feature_map[:, :, h_start:h_end, w_start:w_end]  # [B, C, block_h, block_w]

                # 块内平均
                block_avg = torch.mean(block, dim=[-2, -1])  # [B, C]
                block_features.append(block_avg)

        # 将所有块特征拼接
        block_features = torch.stack(block_features, dim=1)  # [B, num_blocks*num_blocks, C]

        # 步骤2: 频率域特征提取
        # 初始化频率分析层（如果不存在）
        if not hasattr(self, 'frequency_transform'):
            # 使用1D卷积作为频率变换（简化的小波变换模拟）
            self.frequency_transform = nn.Conv1d(
                in_channels=C,
                out_channels=num_endm * num_queries_per_endm,
                kernel_size=3,
                padding=1,
                groups=1,
                device=device
            )

        # 为每个块提取频率特征
        freq_features_list = []
        for b in range(B):
            # [num_blocks*num_blocks, C] -> [C, num_blocks*num_blocks]
            block_feats_b = block_features[b].transpose(0, 1).unsqueeze(0)  # [1, C, num_blocks*num_blocks]

            # 应用频率变换
            freq_feats = self.frequency_transform(
                block_feats_b)  # [1, num_endm*num_queries_per_endm, num_blocks*num_blocks]
            freq_feats = freq_feats.view(1, num_endm, num_queries_per_endm,
                                         -1)  # [1, num_endm, num_queries_per_endm, num_blocks*num_blocks]

            # 对空间维度求平均
            freq_feats = torch.mean(freq_feats, dim=-1)  # [1, num_endm, num_queries_per_endm]
            freq_features_list.append(freq_feats)

        # 合并所有批次的频率特征
        freq_features = torch.cat(freq_features_list, dim=0)  # [B, num_endm, num_queries_per_endm]

        # 步骤3: 初始化查询嵌入（如果不存在）
        if not hasattr(self, 'query_embed'):
            self.query_embed = nn.Embedding(
                num_endm * num_queries_per_endm,
                C,  # 输出维度为光谱波段数
                device=device
            )

        # 分割查询嵌入
        query_embed_weight_split = torch.chunk(self.query_embed.weight, num_endm, dim=0)
        query_embed_weight_split = torch.stack(query_embed_weight_split)  # [num_endm, num_queries_per_endm, C]

        # 步骤4: 初始化注意力权重（如果不存在）
        if not hasattr(self, 'weights'):
            # 初始化权重矩阵 - 这里我们使用特征图提取的频率信息初始化
            self.weights = nn.Parameter(
                torch.randn(num_endm, num_queries_per_endm, device=device),
                requires_grad=True
            )

        # 步骤5: 频率特征调制权重
        # 先对批次维度求平均，获得更稳定的特征表示
        avg_freq_features = torch.mean(freq_features, dim=0)  # [num_endm, num_queries_per_endm]

        # 结合预设权重和频率特征（按一定比例）
        alpha = 0.01  # 特征影响因子
        modulated_weights = self.weights * (1 + alpha * avg_freq_features)

        # 步骤6: 计算注意力权重
        attn_weights = F.softmax(modulated_weights, dim=1)  # [num_endm, num_queries_per_endm]

        # 步骤7: 应用注意力权重生成端元
        endmember_get = attn_weights.unsqueeze(-1) * query_embed_weight_split  # [num_endm, num_queries_per_endm, C]
        endmember_get = torch.sum(endmember_get, dim=1)  # [num_endm, C]

        return endmember_get

    def get_endmember(self):
        # 分割查询嵌入
        query_embed_weight_split = torch.chunk(self.query_embed.weight, self.num_endm, dim=0)
        query_embed_weight_split = torch.stack(
            query_embed_weight_split)  # [num_endm, num_queries_per_endm, num_band]

        # 计算注意力权重
        attn_weights = F.softmax(self.weights, dim=1)  # [num_endm, num_queries_times]

        # 应用注意力权重
        endmember_get = attn_weights.unsqueeze(
            -1) * query_embed_weight_split  # [num_endm, num_queries_times, num_band]
        endmember_get = torch.sum(endmember_get, dim=1)  # [num_endm, num_band]

        return endmember_get



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_endmember = 4
    num_band = 285
    rows = [110, 110]
    model = FACM(rows[0],rows[1],num_band).to(device)
    # print(model)
    # summary(model, [num_band, 100, 100])
    input_data = torch.randn(1, num_band, rows[0], rows[1]).to(device)
    recon_linear, abun_get, endm_get = model(input_data)
    print(recon_linear.shape)
    print(abun_get.shape)
    print(endm_get.shape)
    input_data = torch.randn(1, num_band, rows[0], rows[1]).to(device)
    flops, params = profile(model, inputs=(input_data,))
    print('params:', params)
    print('flops:', flops)

