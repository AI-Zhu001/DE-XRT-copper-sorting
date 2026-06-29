# Robust Dual-Energy X-ray Copper Ore Sorting via Log-Domain Contrast Imaging and Soft Adaptive Attenuation Compensation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper:

**"Robust Dual-Energy X-ray Copper Ore Sorting via Log-Domain Contrast Imaging and Soft Adaptive Attenuation Compensation"**  
*Zhu Zhi-yong, He Jian-feng, Wang Wen, Cai You-bin, Wang Xue-yuan, Xia Fei, Nie Feng-jun, Li Wei-dong, Zhong Guo-yun, Ye Zhi-Xiang, Diao Fan*

## 📝 Overview

This work proposes a lightweight test-time framework to handle attenuation-response variations in Dual‑Energy X‑ray Transmission (DE‑XRT) copper ore sorting. The key contributions are:

- **Physical Contrast Imaging (PCI)**: A log-domain three-channel representation that encodes dual-energy attenuation contrast.
- **Soft Adaptive Attenuation Compensation (Soft APC)**: A sample-level, dead-zone-controlled compensation mechanism that corrects response drift at inference time without retraining.
- **Demonstrated Robustness**: Improves overall accuracy from 70.41% to 84.70% under extreme negative offset, and from 43.05% to 75.47% under extreme positive offset.
- **Real‑Time Ready**: Achieves 13.65 ms per‑frame inference on an 8‑thread CPU.

## 📊 Dataset

The dataset used in this study is available at Figshare:  
[https://doi.org/10.6084/m9.figshare.32150047](https://doi.org/10.6084/m9.figshare.32150047)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- Pillow
- tqdm
- scipy
- pandas

### Installation
Clone the repository:
```bash
git clone https://github.com/AI-Zhu001/DE-XRT-thickness-regularization.git
cd DE-XRT-thickness-regularization
