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
from scipy.stats import chi2


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
def build_baseline_input(h, l):
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
    x = torch.cat([h_c, l_c, d_c], dim=1)

    return x


# ============================================================
# 4. McNemar test
# ============================================================
def mcnemar_test(b, c):
    """
    b: Baseline correct, Proposed wrong
    c: Baseline wrong, Proposed correct

    Uses exact binomial test approximation when b+c is small,
    otherwise uses continuity-corrected chi-square test.
    Here we implement continuity-corrected chi-square because scipy
    binomtest may not exist in older environments.
    """
    n = b + c

    if n == 0:
        return 1.0, 0.0

    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = 1 - chi2.cdf(statistic, df=1)

    return p_value, statistic


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
    records = []

    for delta in deltas:
        print(f"\nEvaluating McNemar test: seed={seed}, delta={delta:+.2f}")

        baseline_correct = []
        proposed_correct = []

        baseline_preds = []
        proposed_preds = []
        targets = []

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
                pred_base = model(x_base).argmax(1).item()

                x_apc = build_soft_apc_input(
                    h_env,
                    l_env,
                    m_ref=-0.5185,
                    dead_zone=0.08,
                    gain=1.1,
                    max_comp=0.25
                )
                pred_prop = model(x_apc).argmax(1).item()

            baseline_preds.append(pred_base)
            proposed_preds.append(pred_prop)
            targets.append(target)

            baseline_correct.append(pred_base == target)
            proposed_correct.append(pred_prop == target)

        baseline_correct = np.asarray(baseline_correct, dtype=bool)
        proposed_correct = np.asarray(proposed_correct, dtype=bool)

        # Contingency table:
        # n00: both wrong
        # n01: baseline wrong, proposed correct
        # n10: baseline correct, proposed wrong
        # n11: both correct
        n11 = int(np.sum(baseline_correct & proposed_correct))
        n10 = int(np.sum(baseline_correct & ~proposed_correct))
        n01 = int(np.sum(~baseline_correct & proposed_correct))
        n00 = int(np.sum(~baseline_correct & ~proposed_correct))

        b = n10
        c = n01

        p_value, statistic = mcnemar_test(b, c)

        baseline_acc = float(np.mean(baseline_correct))
        proposed_acc = float(np.mean(proposed_correct))
        acc_gain = proposed_acc - baseline_acc

        records.append({
            "Seed": seed,
            "Delta": delta,
            "N": len(targets),
            "BothCorrect_n11": n11,
            "BaselineCorrect_ProposedWrong_n10": n10,
            "BaselineWrong_ProposedCorrect_n01": n01,
            "BothWrong_n00": n00,
            "b": b,
            "c": c,
            "McNemarStatistic": statistic,
            "PValue": p_value,
            "Significant_p_lt_0_05": p_value < 0.05,
            "BaselineAcc": baseline_acc,
            "ProposedAcc": proposed_acc,
            "AccGain": acc_gain
        })

    out_path = f"results/MCNEMAR_TEST_seed{seed}.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return pd.DataFrame(records)


# ============================================================
# 6. Summary
# ============================================================
def summarize(all_df):
    out_all = "results/MCNEMAR_TEST_ALL_SEEDS.csv"
    all_df.to_csv(out_all, index=False)

    summary_records = []

    for delta, df_delta in all_df.groupby("Delta"):
        summary_records.append({
            "Delta": delta,
            "BaselineAcc_mean_percent": df_delta["BaselineAcc"].mean() * 100,
            "BaselineAcc_std_percent": df_delta["BaselineAcc"].std() * 100,
            "ProposedAcc_mean_percent": df_delta["ProposedAcc"].mean() * 100,
            "ProposedAcc_std_percent": df_delta["ProposedAcc"].std() * 100,
            "AccGain_mean_percent": df_delta["AccGain"].mean() * 100,
            "AccGain_std_percent": df_delta["AccGain"].std() * 100,
            "PValue_mean": df_delta["PValue"].mean(),
            "PValue_max": df_delta["PValue"].max(),
            "Significant_all_3_seeds": bool(df_delta["Significant_p_lt_0_05"].all()),
            "Mean_b": df_delta["b"].mean(),
            "Mean_c": df_delta["c"].mean()
        })

    summary = pd.DataFrame(summary_records)

    summary["BaselineAcc_mean_std"] = summary.apply(
        lambda r: f"{r['BaselineAcc_mean_percent']:.2f} ± {r['BaselineAcc_std_percent']:.2f}",
        axis=1
    )

    summary["ProposedAcc_mean_std"] = summary.apply(
        lambda r: f"{r['ProposedAcc_mean_percent']:.2f} ± {r['ProposedAcc_std_percent']:.2f}",
        axis=1
    )

    summary["AccGain_mean_std"] = summary.apply(
        lambda r: f"{r['AccGain_mean_percent']:.2f} ± {r['AccGain_std_percent']:.2f}",
        axis=1
    )

    out_summary = "results/MCNEMAR_TEST_MEAN_SUMMARY.csv"
    summary.to_csv(out_summary, index=False)

    display_cols = [
        "Delta",
        "BaselineAcc_mean_std",
        "ProposedAcc_mean_std",
        "AccGain_mean_std",
        "PValue_mean",
        "PValue_max",
        "Significant_all_3_seeds",
        "Mean_b",
        "Mean_c"
    ]

    print("\n" + "=" * 120)
    print("McNemar test summary over 3 seeds")
    print("=" * 120)
    print(summary[display_cols].to_string(index=False))

    print(f"\nSaved all-seed results: {out_all}")
    print(f"Saved summary: {out_summary}")


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
    summarize(all_df)