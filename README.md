cat << 'EOF' > README.md
# Robust Dual-Energy X-ray Copper Ore Sorting via Log-Domain Contrast Imaging and Soft Adaptive Attenuation Compensation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper:

**"Robust Dual-Energy X-ray Copper Ore Sorting via Log-Domain Contrast Imaging and Soft Adaptive Attenuation Compensation"**  
*Zhu Zhi-yong, He Jian-feng, Wang Wen, Cai You-bin, Wang Xue-yuan, Xia Fei, Nie Feng-jun, Li Wei-dong, Zhong Guo-yun, Ye Zhi-Xiang, Diao Fan*

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

https://doi.org/10.6084/m9.figshare.32150047

---

## 📦 Installation

git clone https://github.com/AI-Zhu001/DE-XRT-thickness-regularization.git
cd DE-XRT-thickness-regularization
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

@article{zhu2026robust,
  title={Robust Dual-Energy X-ray Copper Ore Sorting via Log-Domain Contrast Imaging and Soft Adaptive Attenuation Compensation},
  author={Zhu, Zhi-yong and He, Jian-feng and Wang, Wen and Cai, You-bin and Wang, Xue-yuan and Xia, Fei and Nie, Feng-jun and Li Wei-dong and Zhong, Guo-yun and Ye, Zhi-Xiang and Diao, Fan},
  journal={International Journal of Minerals, Metallurgy and Materials},
  year={2026}
}

---

## 📧 Contact

Jianfeng He – hjf_10@yeah.net

---

## 📜 License

MIT License
EOF
