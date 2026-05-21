import numpy as np
import os
from matplotlib import pyplot as plt


def plot_abundance(ground_truth, estimated, em, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 6), dpi=150)
    if ground_truth is not None:
        for i in range(em):
            plt.subplot(2, em, i + 1)
            plt.imshow(ground_truth[:, :, i], cmap='jet')
            plt.title(f'GT {i+1}')
        for i in range(em):
            plt.subplot(2, em, em + i + 1)
            plt.imshow(estimated[:, :, i], cmap='jet')
            plt.title(f'Est {i+1}')
        plt.tight_layout()
    else:
        for i in range(em):
            plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
            plt.imshow(estimated[:, :, i], cmap='jet')
            plt.title(f'Est {i+1}')
        plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "abundances.png"))
    plt.show()


def plot_endmembers(target, pred, em, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        plt.plot(pred[:, i]/max(pred[:, i] + 1e-8), label="Extracted")  # 防止除零
        if target is not None:
            plt.plot(target[:, i]/max(target[:, i] + 1e-8), label="GT")
        plt.legend(loc="upper left")
        plt.title(f"EM {i+1}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "endmembers_norm.png"))
    plt.show()
