```markdown
# Hyperspectral Unmixing Using Frequency-Adaptive Convolutional-Mamba Network (FACM)

## ⚙️ Prerequisites

* Python 3.8+
* PyTorch (CUDA recommended)
* `scipy`, `numpy`, `matplotlib`, `einops`, `thop`

Install the required dependencies:
```bash
pip install torch scipy numpy matplotlib einops thop

```

---

## 📂 Data Preparation

The code supports multiple benchmark hyperspectral datasets. Please place the `.mat` data files in your designated dataset directory.
Supported datasets include:

* Simulated Data 


* Jasper Ridge 


* Urban 


* Apex 


* Orchard 


* Cuprite 



---

## 🚀 Quick Start

To train and evaluate the FACM model, simply run `main.py`. The framework is fully unsupervised and relies solely on self-supervised constraints derived from the input data.

### Basic Usage

Run on the Jasper Ridge dataset (JR) with default parameters:

```bash
python main.py -d JR

```

### Advanced Usage

Train on the Urban dataset (UR) with custom settings for epochs and loss weights:

```bash
python main.py -d UR -e 500 -l 3e-3 --weight_sad 1.2 --weight_fft 1e-5

```

(Note: Hyperparameters such as epochs, learning rates, and loss weights are automatically tuned for different datasets as detailed in the paper's experimental setup.)

---

## 📖 Citation

If you find our code or methodology helpful for your research, please cite our work:

```bibtex
@article{zhao2026hyperspectral,
  title={Hyperspectral Unmixing Using Frequency-Adaptive Convolutional-Mamba Network},
  author={Zhao, Zhuoyi and Xu, Xiang and Deng, Chuiyi and Guo, Yanyin and Li, Junwei and Plaza, Antonio},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026},
  publisher={IEEE}
}

```

## ✉️ Contact

For any questions regarding the code or the paper, please open an issue in this repository or contact the authors.

```
wechat:flyzzie
mail:flyzzie@zju.edu.cn
```
