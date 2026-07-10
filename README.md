cat << 'EOF' > README.md
# Soft Adaptive Attenuation Compensation for Robust Dual-Energy X-ray Copper Ore Sorting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper:

**"Soft Adaptive Attenuation Compensation for Robust Dual-Energy X-ray Copper Ore Sorting"**  
*Zhi-yong Zhu, Jian-feng He, Xue-yuan Wang, Fei Xia, Feng-jun Nie, Wen Wang, Yang-hui Zou, Wei-dong Li, Guo-yun Zhong, Zhi-xiang Ye, Fan Diao*

---

## 📝 Overview

This work proposes a lightweight test-time framework for Dual-Energy X-ray Transmission (DE-XRT) copper ore sorting.

### Key Contributions

- **Physical Contrast Imaging (PCI)**: Log-domain three-channel representation encoding dual-energy attenuation contrast.
- **Soft Adaptive Attenuation Compensation (Soft APC)**: Sample-level correction without retraining.
- **Robustness Improvement**: 70.41% → 84.70% (negative offset), 43.05% → 75.47% (positive offset).
- **Real-Time Performance**: 13.65 ms per-frame inference on 8-thread CPU.

---

## 📊 Dataset

The original DE-XRT copper ore image dataset is not publicly available due to data-use restrictions associated with the industrial ore samples.

The repository contains code only and does not include the original high- and low-energy X-ray images.

---

## 📦 Installation

git clone https://github.com/AI-Zhu001/DE-XRT-copper-sorting.git
cd DE-XRT-copper-sorting
pip install -r requirements.txt

---

## 🏃 How to Run

python compute_m_ref_from_training.py

python train_proposed_pci_seeded.py --seed 42

python train_highlow_seeded.py --seed 42
python train_cbal2_sota_seeded.py --seed 42
python train_mobilenetv3_seeded.py --seed 42
python train_efficientnet_b0_seeded.py --seed 42

python eval_five_way_test_clip_seeded.py --seed 42

python eval_pci_input_ablation_seeded.py
python eval_ablation_test_seeded.py
python eval_deadzone_sensitivity_test_seeded.py

python eval_sensor_perturbation_test_seeded.py

python eval_mcnemar_test_seeded.py

python eval_complexity_latency.py --device cpu --threads 8

python summarize_three_seed_results.py

---

## 📁 Repository Structure

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

---

## 📜 Citation

Citation information will be updated after publication.

---

## 📧 Contact

Jianfeng He – hjf_10@yeah.net

---

## 📜 License

MIT License
EOF
