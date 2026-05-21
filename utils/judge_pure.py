import scipy.io as sio
import numpy as np

mat_hsi = sio.loadmat('/data/zzy/HSIU/simulate/30db/data_ex4.mat')
data_aban = mat_hsi['alphas']

# 找出正好等于1的位置
positions = np.where(data_aban == 1.0)

if len(positions[0]) > 0:
    print(f"丰度正好等于1.0的数量: {len(positions[0])}")
    print(f"\n位置列表:")
    for idx in range(len(positions[0])):
        pos = tuple(p[idx] for p in positions)
        print(f"  位置 {pos}: 丰度值 = {data_aban[pos]}")
else:
    print("没有找到丰度正好等于1.0的像元")