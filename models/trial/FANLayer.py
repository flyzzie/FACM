import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True).
            There is almost no difference between bias and non-bias in our experiments.
    """

    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()

        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms

        # Linear transformation for the p component (for cosine and sine parts)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)

        # Linear transformation for the g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else lambda x: x

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """

        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))

        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)

        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)

        return output

    # 测试函数


def test_fan_layer():
    # 设置随机种子以保证结果可重现
    torch.manual_seed(42)
    np.random.seed(42)

    # 测试参数
    batch_size = 4
    input_dim = 16
    output_dim = 32
    p_ratio = 0.25

    # 创建一个FANLayer实例
    fan = FANLayer(input_dim, output_dim, p_ratio=p_ratio, activation='gelu', use_p_bias=True)

    # 生成随机输入
    x = torch.randn(batch_size, input_dim)

    # 打印输入
    print(f"Input shape: {x.shape}")
    print("Sample input:\n", x[0].detach().numpy())

    # 将输入传递给FANLayer
    output = fan(x)

    # 打印输出
    print(f"\nOutput shape: {output.shape}")
    print("Sample output:\n", output[0].detach().numpy())

    # 计算各部分的维度
    p_dim = int(output_dim * p_ratio)
    g_dim = output_dim - 2 * p_dim

    # 分离并检查各组件
    cos_p = output[:, :p_dim]
    sin_p = output[:, p_dim:2 * p_dim]
    g_part = output[:, 2 * p_dim:]

    print(f"\nComponent dimensions:")
    print(f"- Cosine part (p_dim): {cos_p.shape} - Expected: {(batch_size, p_dim)}")
    print(f"- Sine part (p_dim): {sin_p.shape} - Expected: {(batch_size, p_dim)}")
    print(f"- G part (g_dim): {g_part.shape} - Expected: {(batch_size, g_dim)}")

    # 验证cos和sin的值在[-1, 1]范围内
    cos_min, cos_max = cos_p.min().item(), cos_p.max().item()
    sin_min, sin_max = sin_p.min().item(), sin_p.max().item()

    print(f"\nValue ranges:")
    print(f"- Cosine values: [{cos_min:.6f}, {cos_max:.6f}] - Expected: [-1, 1]")
    print(f"- Sine values: [{sin_min:.6f}, {sin_max:.6f}] - Expected: [-1, 1]")

    # 计算参数数量
    total_params = sum(p.numel() for p in fan.parameters())
    print(f"\nTotal parameters: {total_params}")

    # 显示p和g的原始变换结果
    with torch.no_grad():
        p_raw = fan.input_linear_p(x)
        g_raw = fan.input_linear_g(x)
        g_activated = fan.activation(g_raw)

    print(f"\nRaw transformations:")
    print(f"- p_raw shape: {p_raw.shape}")
    print(f"- p_raw sample: {p_raw[0][:5].numpy()}")  # 显示前5个值
    print(f"- g_raw shape: {g_raw.shape}")
    print(f"- g_raw sample: {g_raw[0][:5].numpy()}")  # 显示前5个值
    print(f"- g_activated sample: {g_activated[0][:5].numpy()}")  # 显示前5个值

    # 可视化正弦和余弦部分
    visualize_fan_transformation(p_raw[0].numpy(), cos_p[0].numpy(), sin_p[0].numpy())

    return fan, x, output


def visualize_fan_transformation(p_raw, cos_values, sin_values):
    """可视化FANLayer中的正弦和余弦变换"""
    plt.figure(figsize=(12, 6))

    # 绘制原始p值和对应的正弦余弦值
    idx = np.arange(len(p_raw))

    plt.subplot(2, 1, 1)
    plt.title("Raw p values vs Cosine and Sine outputs")
    plt.scatter(idx, p_raw, color='blue', label='Raw p values')
    plt.scatter(idx, cos_values, color='red', label='cos(p)')
    plt.scatter(idx, sin_values, color='green', label='sin(p)')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 展示正弦余弦变换的周期特性
    plt.subplot(2, 1, 2)
    x_range = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    plt.plot(x_range, np.cos(x_range), 'r-', label='cos(x)')
    plt.plot(x_range, np.sin(x_range), 'g-', label='sin(x)')

    # 在图上标记原始p值的位置
    for i, p in enumerate(p_raw):
        plt.axvline(x=p, color='blue', linestyle='--', alpha=0.3)
        plt.text(p, 1.1, f"{i}", fontsize=8, ha='center')

    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title("Sine and Cosine functions with p values marked")
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-1.2, 1.2)
    plt.legend()

    plt.tight_layout()
    plt.savefig('fan_transformation.png')
    plt.show()


# 测试不同参数配置
def test_fan_with_different_configs():
    print("=== Testing FANLayer with different configurations ===")

    configs = [
        {"input_dim": 10, "output_dim": 20, "p_ratio": 0.2, "activation": 'relu', "use_p_bias": True},
        {"input_dim": 16, "output_dim": 32, "p_ratio": 0.25, "activation": 'gelu', "use_p_bias": False},
        {"input_dim": 64, "output_dim": 128, "p_ratio": 0.3, "activation": 'silu', "use_p_bias": True},
        {"input_dim": 256, "output_dim": 512, "p_ratio": 0.4, "activation": 'gelu', "use_p_bias": True}
    ]

    for i, config in enumerate(configs):
        print(f"\n--- Configuration {i + 1} ---")
        print(f"Input dim: {config['input_dim']}, Output dim: {config['output_dim']}")
        print(f"p_ratio: {config['p_ratio']}, Activation: {config['activation']}, Use p bias: {config['use_p_bias']}")

        fan = FANLayer(**config)
        x = torch.randn(2, config['input_dim'])

        output = fan(x)
        p_dim = int(config['output_dim'] * config['p_ratio'])

        print(f"Output shape: {output.shape}")
        print(f"Cosine part: {p_dim} dims, Sine part: {p_dim} dims, G part: {config['output_dim'] - 2 * p_dim} dims")

        # 检查值范围
        cos_part = output[:, :p_dim]
        sin_part = output[:, p_dim:2 * p_dim]

        print(f"Cosine range: [{cos_part.min().item():.4f}, {cos_part.max().item():.4f}]")
        print(f"Sine range: [{sin_part.min().item():.4f}, {sin_part.max().item():.4f}]")

        # 计算参数数量
        total_params = sum(p.numel() for p in fan.parameters())
        print(f"Total parameters: {total_params}")


if __name__ == "__main__":
    print("Testing FANLayer implementation...\n")
    fan, x, output = test_fan_layer()

    print("\n\nTesting FANLayer with various configurations...\n")
    test_fan_with_different_configs()