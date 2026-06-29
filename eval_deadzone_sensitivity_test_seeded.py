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
# 2. Data index
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
# 3. APC input builder
# ============================================================
def build_apc_input(h, l, dead_zone, gain=1.1, max_comp=0.25, m_ref=-0.5185):
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
    x = torch.cat([h_c, l_c, d_c], dim=1)

    return x


def build_baseline_input(h, l):
    eps = 1e-6
    d = torch.log(h + eps) - torch.log(l + eps)
    return torch.cat([h, l, d], dim=1)


# ============================================================
# 4. Metrics
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
# 5. One seed evaluation
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

    deltas = [-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25]
    dead_zones = [0.00, 0.04, 0.08, 0.12, 0.16]

    records = []

    for delta in deltas:
        print(f"\nEvaluating seed={seed}, delta={delta:+.2f}")

        pred_dict = {"Baseline": []}
        for dz in dead_zones:
            pred_dict[f"DZ={dz:.2f}"] = []

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
            h_env = torch.clamp(torch.exp(torch.log(h_raw + eps) + delta), 0, 1)
            l_env = torch.clamp(torch.exp(torch.log(l_raw + eps) + delta), 0, 1)

            with torch.no_grad():
                x_base = build_baseline_input(h_env, l_env)
                pred_dict["Baseline"].append(model(x_base).argmax(1).item())

                for dz in dead_zones:
                    x_apc = build_apc_input(
                        h_env,
                        l_env,
                        dead_zone=dz,
                        gain=1.1,
                        max_comp=0.25,
                        m_ref=-0.5185
                    )
                    pred_dict[f"DZ={dz:.2f}"].append(model(x_apc).argmax(1).item())

            targets.append(target)
            groups.append(get_group(h_raw, l_raw))

        for variant, preds in pred_dict.items():
            oa, wga, f1, macro_recall = compute_metrics(targets, preds, groups)

            records.append({
                "Seed": seed,
                "Delta": delta,
                "Variant": variant,
                "OA": oa,
                "WGA": wga,
                "F1": f1,
                "MacroRecall": macro_recall
            })

    out_path = f"results/DEADZONE_SENSITIVITY_TEST_seed{seed}.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return pd.DataFrame(records)


# ============================================================
# 6. Summary
# ============================================================
def summarize(all_df):
    metric_cols = ["OA", "WGA", "F1", "MacroRecall"]

    summary = (
        all_df
        .groupby(["Delta", "Variant"])[metric_cols]
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

    # Robustness summary: average of extreme shifts and center performance
    numeric = summary.copy()
    numeric["OA_mean_percent"] = numeric["OA_mean"] * 100
    numeric["WGA_mean_percent"] = numeric["WGA_mean"] * 100
    numeric["F1_mean_percent"] = numeric["F1_mean"] * 100
    numeric["MacroRecall_mean_percent"] = numeric["MacroRecall_mean"] * 100

    extreme = numeric[numeric["Delta"].isin([-0.25, 0.25])]
    extreme_summary = (
        extreme
        .groupby("Variant")[["OA_mean_percent", "WGA_mean_percent", "F1_mean_percent", "MacroRecall_mean_percent"]]
        .mean()
        .reset_index()
    )

    center = numeric[numeric["Delta"] == 0.0][
        ["Variant", "OA_mean_percent", "WGA_mean_percent", "F1_mean_percent", "MacroRecall_mean_percent"]
    ].copy()
    center = center.rename(columns={
        "OA_mean_percent": "Center_OA",
        "WGA_mean_percent": "Center_WGA",
        "F1_mean_percent": "Center_F1",
        "MacroRecall_mean_percent": "Center_MacroRecall"
    })

    extreme_summary = extreme_summary.rename(columns={
        "OA_mean_percent": "ExtremeAvg_OA",
        "WGA_mean_percent": "ExtremeAvg_WGA",
        "F1_mean_percent": "ExtremeAvg_F1",
        "MacroRecall_mean_percent": "ExtremeAvg_MacroRecall"
    })

    overview = extreme_summary.merge(center, on="Variant", how="left")

    summary_out = "results/DEADZONE_SENSITIVITY_TEST_MEAN_STD.csv"
    overview_out = "results/DEADZONE_SENSITIVITY_OVERVIEW.csv"

    summary.to_csv(summary_out, index=False)
    overview.to_csv(overview_out, index=False)

    display_cols = [
        "Delta",
        "Variant",
        "OA_percent_mean_std",
        "WGA_percent_mean_std",
        "F1_percent_mean_std",
        "MacroRecall_percent_mean_std"
    ]

    print("\n" + "=" * 120)
    print("Dead-zone sensitivity summary, mean ± std (%):")
    print("=" * 120)
    print(summary[display_cols].to_string(index=False))

    print("\n" + "=" * 120)
    print("Overview: extreme average and center performance (%):")
    print("=" * 120)
    print(overview.to_string(index=False))

    print(f"\nSaved summary: {summary_out}")
    print(f"Saved overview: {overview_out}")


# ============================================================
# 7. Entry
# ============================================================
if __name__ == "__main__":
    seeds = [42, 43, 44]

    all_results = []
    for seed in seeds:
        df_seed = evaluate_one_seed(seed)
        all_results.append(df_seed)

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv("results/DEADZONE_SENSITIVITY_TEST_ALL_SEEDS.csv", index=False)

    summarize(all_df)