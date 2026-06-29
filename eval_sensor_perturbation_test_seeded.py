import os
import random
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, recall_score


# ============================================================
# 0. Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 1. Model
# ============================================================
class TICL_PCI_Net_Pro(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18()
        self.resnet.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.classifier(self.resnet(self.normalize(x)))


# ============================================================
# 2. File index
# ============================================================
def build_index(data_root):
    f_map = {}

    for p in Path(data_root).rglob("*"):
        if p.suffix.lower() in [".jpg", ".png"]:
            name = p.name.replace(" ", "").lower()

            if "high" in str(p).lower():
                tag = "high"
            elif "low" in str(p).lower():
                tag = "low"
            else:
                continue

            f_map[(tag, name)] = str(p)

    print(f"Indexed image files: {len(f_map)}")
    return f_map


# ============================================================
# 3. Input builders
# ============================================================
def build_pci_input(h, l):
    eps = 1e-6
    d = torch.log(h + eps) - torch.log(l + eps)
    return torch.cat([h, l, d], dim=1)


def build_soft_apc_input(
    h,
    l,
    m_ref=-0.5185,
    dead_zone=0.08,
    gain=1.1,
    max_comp=0.25
):
    eps = 1e-6

    log_h = torch.log(h + eps)
    log_l = torch.log(l + eps)

    m_curr = 0.5 * (log_h + log_l).mean().item()
    diff = m_ref - m_curr
    abs_d = abs(diff)

    if abs_d > dead_zone:
        d_comp = np.sign(diff) * (abs_d - dead_zone) * gain
    else:
        d_comp = 0.0

    d_comp = float(np.clip(d_comp, -max_comp, max_comp))

    h_c = torch.clamp(torch.exp(log_h + d_comp), 0, 1)
    l_c = torch.clamp(torch.exp(log_l + d_comp), 0, 1)

    d_c = torch.log(h_c + eps) - torch.log(l_c + eps)

    return torch.cat([h_c, l_c, d_c], dim=1)


# ============================================================
# 4. Sensor perturbations
# ============================================================
def apply_sensor_perturbation(h, l, perturbation_name):
    """
    h, l: tensors in [0, 1], shape [1, 1, H, W]
    Perturbations are applied to both high- and low-energy channels.
    """

    if perturbation_name == "Clean":
        h_p, l_p = h, l

    elif perturbation_name == "Gaussian_0.01":
        h_p = h + torch.randn_like(h) * 0.01
        l_p = l + torch.randn_like(l) * 0.01

    elif perturbation_name == "Gaussian_0.03":
        h_p = h + torch.randn_like(h) * 0.03
        l_p = l + torch.randn_like(l) * 0.03

    elif perturbation_name == "Gain_0.95":
        h_p = h * 0.95
        l_p = l * 0.95

    elif perturbation_name == "Gain_1.05":
        h_p = h * 1.05
        l_p = l * 1.05

    elif perturbation_name == "Bias_0.02":
        h_p = h + 0.02
        l_p = l + 0.02

    elif perturbation_name == "Mixed":
        # Mild combined perturbation: gain drift + small additive noise
        h_p = h * 1.03 + torch.randn_like(h) * 0.01
        l_p = l * 0.97 + torch.randn_like(l) * 0.01

    else:
        raise ValueError(f"Unknown perturbation: {perturbation_name}")

    h_p = torch.clamp(h_p, 0, 1)
    l_p = torch.clamp(l_p, 0, 1)

    return h_p, l_p


# ============================================================
# 5. Metrics
# ============================================================
def get_group(h_raw, l_raw):
    eps = 1e-6
    m_orig = 0.5 * (
        torch.log(h_raw + eps) + torch.log(l_raw + eps)
    ).mean().item()

    if m_orig > -0.45:
        return "Thin"
    if m_orig < -0.65:
        return "Thick"
    return "Medium"


def compute_metrics(y_true, y_pred, groups):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    oa = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")

    group_accs = []
    for g in np.unique(groups):
        mask = groups == g
        if np.any(mask):
            group_accs.append(accuracy_score(y_true[mask], y_pred[mask]))

    wga = min(group_accs) if group_accs else 0.0

    return oa, wga, f1, macro_recall


# ============================================================
# 6. One seed evaluation
# ============================================================
def evaluate_one_seed(seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Seed: {seed}")

    csv_path = "/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"
    data_root = "/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集"

    weight_path = f"weights/best_model_FINAL_seed{seed}.pth"
    if not Path(weight_path).exists():
        raise FileNotFoundError(f"Missing weight file: {weight_path}")

    os.makedirs("results", exist_ok=True)

    model = TICL_PCI_Net_Pro().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.eval()

    f_map = build_index(data_root)

    df = pd.read_csv(csv_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor()
    ])

    deltas = [-0.25, 0.00, 0.25]

    perturbations = [
        "Clean",
        "Gaussian_0.01",
        "Gaussian_0.03",
        "Gain_0.95",
        "Gain_1.05",
        "Bias_0.02",
        "Mixed"
    ]

    records = []

    for delta in deltas:
        for perturb in perturbations:
            print(f"\nEvaluating seed={seed}, delta={delta:+.2f}, perturbation={perturb}")

            preds_baseline = []
            preds_proposed = []
            targets = []
            groups = []

            for _, row in tqdm(test_df.iterrows(), total=len(test_df), leave=False):
                h_name = PureWindowsPath(row["high_path"]).name.replace(" ", "").lower()
                l_name = PureWindowsPath(row["low_path"]).name.replace(" ", "").lower()

                h_path = f_map.get(("high", h_name))
                l_path = f_map.get(("low", l_name))

                if h_path is None or l_path is None:
                    continue

                h_raw = transform(Image.open(h_path).convert("L")).unsqueeze(0).to(device)
                l_raw = transform(Image.open(l_path).convert("L")).unsqueeze(0).to(device)

                target = int(row["label"])

                eps = 1e-6

                # Thickness-equivalent shift
                h_env = torch.clamp(torch.exp(torch.log(h_raw + eps) + delta), 0, 1)
                l_env = torch.clamp(torch.exp(torch.log(l_raw + eps) + delta), 0, 1)

                # Additional sensor perturbation
                h_pert, l_pert = apply_sensor_perturbation(h_env, l_env, perturb)

                with torch.no_grad():
                    x_base = build_pci_input(h_pert, l_pert)
                    pred_base = model(x_base).argmax(1).item()

                    x_apc = build_soft_apc_input(
                        h_pert,
                        l_pert,
                        m_ref=-0.5185,
                        dead_zone=0.08,
                        gain=1.1,
                        max_comp=0.25
                    )
                    pred_prop = model(x_apc).argmax(1).item()

                preds_baseline.append(pred_base)
                preds_proposed.append(pred_prop)
                targets.append(target)
                groups.append(get_group(h_raw, l_raw))

            for model_name, pred_list in [
                ("Baseline", preds_baseline),
                ("Proposed", preds_proposed)
            ]:
                oa, wga, f1, macro_recall = compute_metrics(
                    targets,
                    pred_list,
                    groups
                )

                records.append({
                    "Seed": seed,
                    "Delta": delta,
                    "Perturbation": perturb,
                    "Model": model_name,
                    "OA": oa,
                    "WGA": wga,
                    "F1": f1,
                    "MacroRecall": macro_recall
                })

    out_path = f"results/SENSOR_PERTURBATION_TEST_seed{seed}.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    return pd.DataFrame(records)


# ============================================================
# 7. Summary
# ============================================================
def summarize(all_df):
    metric_cols = ["OA", "WGA", "F1", "MacroRecall"]

    summary = (
        all_df
        .groupby(["Delta", "Perturbation", "Model"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        summary[f"{metric}_percent_mean_std"] = summary.apply(
            lambda r: f"{r[mean_col] * 100:.2f} ± {r[std_col] * 100:.2f}",
            axis=1
        )

    # Compute Proposed - Baseline gain under each condition
    gain_records = []
    for (delta, perturb), group in all_df.groupby(["Delta", "Perturbation"]):
        base = group[group["Model"] == "Baseline"]
        prop = group[group["Model"] == "Proposed"]

        for metric in metric_cols:
            base_vals = base.sort_values("Seed")[metric].values
            prop_vals = prop.sort_values("Seed")[metric].values

            if len(base_vals) == len(prop_vals) and len(base_vals) > 0:
                gain_vals = (prop_vals - base_vals) * 100
                gain_records.append({
                    "Delta": delta,
                    "Perturbation": perturb,
                    "Metric": metric,
                    "Gain_mean_percent": np.mean(gain_vals),
                    "Gain_std_percent": np.std(gain_vals, ddof=1)
                })

    gain_df = pd.DataFrame(gain_records)

    summary_out = "results/SENSOR_PERTURBATION_TEST_MEAN_STD.csv"
    gain_out = "results/SENSOR_PERTURBATION_GAIN_SUMMARY.csv"

    summary.to_csv(summary_out, index=False)
    gain_df.to_csv(gain_out, index=False)

    display_cols = [
        "Delta",
        "Perturbation",
        "Model",
        "OA_percent_mean_std",
        "WGA_percent_mean_std",
        "F1_percent_mean_std",
        "MacroRecall_percent_mean_std"
    ]

    print("\n" + "=" * 140)
    print("Sensor perturbation summary, mean ± std (%):")
    print("=" * 140)
    print(summary[display_cols].to_string(index=False))

    print("\n" + "=" * 140)
    print("Proposed - Baseline gain summary (%):")
    print("=" * 140)
    print(gain_df.to_string(index=False))

    print(f"\nSaved summary: {summary_out}")
    print(f"Saved gain summary: {gain_out}")


# ============================================================
# 8. Entry
# ============================================================
if __name__ == "__main__":
    seeds = [42, 43, 44]

    all_results = []
    for seed in seeds:
        df_seed = evaluate_one_seed(seed)
        all_results.append(df_seed)

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv("results/SENSOR_PERTURBATION_TEST_ALL_SEEDS.csv", index=False)

    summarize(all_df)