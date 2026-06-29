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

Install dependencies:
pip install -r requirements.txt

🏃 How to Run
All experiments are run with three random seeds (42, 43, 44). The commands below show an example for seed 42.

1️⃣ Compute training‑set reference response (required before any evaluation)
bash
python compute_m_ref_from_training.py
2️⃣ Train the proposed PCI model (ResNet18)
bash
python train_proposed_pci_seeded.py --seed 42
3️⃣ Train baseline and comparison models
bash
python train_highlow_seeded.py --seed 42          # High+Low baseline
python train_cbal2_sota_seeded.py --seed 42       # CBAL2‑Net
python train_mobilenetv3_seeded.py --seed 42      # MobileNetV3‑Small
python train_efficientnet_b0_seeded.py --seed 42  # EfficientNet‑B0
4️⃣ Evaluate all models under attenuation‑equivalent offsets
bash
python eval_five_way_test_clip_seeded.py --seed 42
5️⃣ Run ablation studies
bash
python eval_pci_input_ablation_seeded.py           # PCI vs High+Low vs PCI+SoftAPC
python eval_ablation_test_seeded.py                # Soft APC component ablation
python eval_deadzone_sensitivity_test_seeded.py    # Dead‑zone sensitivity
6️⃣ Run sensor perturbation tests
bash
python eval_sensor_perturbation_test_seeded.py
7️⃣ Run McNemar statistical test
bash
python eval_mcnemar_test_seeded.py
8️⃣ Measure model complexity and inference latency
bash
python eval_complexity_latency.py --device cpu --threads 8
9️⃣ Generate summary tables (mean ± std over three seeds)
bash
python summarize_three_seed_results.py
📁 Repository Structure
text
DE-XRT-thickness-regularization/
├── README.md
├── LICENSE
├── requirements.txt
├── compute_m_ref_from_training.py
├── train_proposed_pci_seeded.py
├── train_highlow_seeded.py
├── train_cbal2_sota_seeded.py
├── train_mobilenetv3_seeded.py
├── train_efficientnet_b0_seeded.py
├── eval_five_way_test_clip_seeded.py
├── eval_pci_input_ablation_seeded.py
├── eval_ablation_test_seeded.py
├── eval_deadzone_sensitivity_test_seeded.py
├── eval_sensor_perturbation_test_seeded.py
├── eval_mcnemar_test_seeded.py
├── eval_complexity_latency.py
└── summarize_three_seed_results.py
📜 Citation
If you find this work useful for your research, please cite our paper:

bibtex
@article{zhu2026robust,
  title={Robust Dual-Energy X-ray Copper Ore Sorting via Log-Domain Contrast Imaging and Soft Adaptive Attenuation Compensation},
  author={Zhu, Zhi-yong and He, Jian-feng and Wang, Wen and Cai, You-bin and Wang, Xue-yuan and Xia, Fei and Nie, Feng-jun and Li, Wei-dong and Zhong, Guo-yun and Ye, Zhi-Xiang and Diao, Fan},
  journal={International Journal of Minerals, Metallurgy and Materials},
  year={2026}
}
📧 Contact
For questions or issues, please contact the corresponding author:
Jianfeng He – hjf_10@yeah.net

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
