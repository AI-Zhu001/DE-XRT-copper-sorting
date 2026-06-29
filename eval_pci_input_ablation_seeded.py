import os
import random
import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, recall_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_index(data_root: str):
    f_map = {}
    for p in Path(data_root).rglob("*"):
        if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
            key = p.name.replace(" ", "").lower()
            path_lower = str(p).lower()
            if "high" in path_lower:
                tag = "high"
            elif "low" in path_lower:
                tag = "low"
            else:
                continue
            f_map[(tag, key)] = str(p)
    print(f"Indexed image files: {len(f_map)}")
    return f_map


class TICL_PCI_Net_Pro(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        return self.classifier(self.resnet(self.normalize(x)))


class TICL_HighLow_Net(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            2,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456],
            std=[0.229, 0.224],
        )

    def forward(self, x):
        return self.classifier(self.resnet(self.normalize(x)))


def build_highlow_input(h, l):
    return torch.cat([h, l], dim=1)


def build_pci_input(h, l):
    eps = 1e-6
    d = torch.log(h + eps) - torch.log(l + eps)
    return torch.cat([h, l, d], dim=1)


def compensate_hl(h, l, m_ref, dead_zone, gain, max_comp):
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
    if max_comp is not None:
        d_comp = float(np.clip(d_comp, -max_comp, max_comp))
    h_c = torch.clamp(torch.exp(log_h + d_comp), 0, 1)
    l_c = torch.clamp(torch.exp(log_l + d_comp), 0, 1)
    return h_c, l_c, float(d_comp), float(m_curr)


def get_group(h_raw, l_raw):
    eps = 1e-6
    m_orig = 0.5 * (torch.log(h_raw + eps) + torch.log(l_raw + eps)).mean().item()
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
    copper_mask = y_true == 1
    copper_recall = np.mean(y_pred[copper_mask] == 1) if np.any(copper_mask) else np.nan
    group_accs = []
    for g in np.unique(groups):
        mask = groups == g
        if np.any(mask):
            group_accs.append(accuracy_score(y_true[mask], y_pred[mask]))
    wga = min(group_accs) if group_accs else 0.0
    return oa, wga, f1, macro_recall, copper_recall


def evaluate_one_seed(args, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Seed: {seed}")

    highlow_weight = os.path.join(args.weight_dir, f"best_model_HIGHLOW_seed{seed}.pth")
    pci_weight = os.path.join(args.weight_dir, f"best_model_FINAL_seed{seed}.pth")
    for path in [highlow_weight, pci_weight]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weight file: {path}")

    highlow_model = TICL_HighLow_Net().to(device)
    highlow_model.load_state_dict(torch.load(highlow_weight, map_location=device), strict=True)
    highlow_model.eval()

    pci_model = TICL_PCI_Net_Pro().to(device)
    pci_model.load_state_dict(torch.load(pci_weight, map_location=device), strict=False)
    pci_model.eval()

    f_map = build_index(args.data_root)
    df = pd.read_csv(args.csv_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test samples: {len(test_df)}")

    to_tensor = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    records = []
    comp_records = []
    deltas = [float(x) for x in args.deltas]

    for delta in deltas:
        print(f"\nEvaluating seed={seed}, delta={delta:+.2f}")
        pred_dict = {
            "High+Low": [],
            "PCI": [],
            "PCI+SoftAPC": [],
        }
        if args.include_highlow_apc:
            pred_dict["High+Low+SoftAPC"] = []

        targets = []
        groups = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), leave=False):
            h_name = PureWindowsPath(row["high_path"]).name.replace(" ", "").lower()
            l_name = PureWindowsPath(row["low_path"]).name.replace(" ", "").lower()
            h_path = f_map.get(("high", h_name))
            l_path = f_map.get(("low", l_name))
            if h_path is None or l_path is None:
                continue

            h_raw = to_tensor(Image.open(h_path).convert("L")).unsqueeze(0).to(device)
            l_raw = to_tensor(Image.open(l_path).convert("L")).unsqueeze(0).to(device)
            target = int(row["label"])

            eps = 1e-6
            h_env = torch.clamp(torch.exp(torch.log(h_raw + eps) + delta), 0, 1)
            l_env = torch.clamp(torch.exp(torch.log(l_raw + eps) + delta), 0, 1)
            h_apc, l_apc, d_comp, m_curr = compensate_hl(
                h_env, l_env,
                m_ref=args.m_ref,
                dead_zone=args.dead_zone,
                gain=args.gain,
                max_comp=args.max_comp,
            )

            with torch.no_grad():
                pred_dict["High+Low"].append(highlow_model(build_highlow_input(h_env, l_env)).argmax(1).item())
                pred_dict["PCI"].append(pci_model(build_pci_input(h_env, l_env)).argmax(1).item())
                pred_dict["PCI+SoftAPC"].append(pci_model(build_pci_input(h_apc, l_apc)).argmax(1).item())
                if args.include_highlow_apc:
                    pred_dict["High+Low+SoftAPC"].append(highlow_model(build_highlow_input(h_apc, l_apc)).argmax(1).item())

            targets.append(target)
            groups.append(get_group(h_raw, l_raw))
            comp_records.append({"Seed": seed, "Delta": delta, "m_curr": m_curr, "d_comp": d_comp})

        for variant, preds in pred_dict.items():
            oa, wga, f1, macro_recall, copper_recall = compute_metrics(targets, preds, groups)
            records.append({
                "Seed": seed,
                "Delta": delta,
                "Variant": variant,
                "OA": oa,
                "WGA": wga,
                "F1": f1,
                "MacroRecall": macro_recall,
                "CopperRecall": copper_recall,
                "N": len(targets),
            })

    out_seed = os.path.join(args.result_dir, f"PCI_INPUT_ABLATION_seed{seed}.csv")
    pd.DataFrame(records).to_csv(out_seed, index=False)
    out_comp = os.path.join(args.result_dir, f"PCI_INPUT_ABLATION_COMPENSATION_seed{seed}.csv")
    pd.DataFrame(comp_records).to_csv(out_comp, index=False)
    print(f"Saved: {out_seed}")
    print(f"Saved: {out_comp}")
    return pd.DataFrame(records)


def summarize(args, all_df):
    metric_cols = ["OA", "WGA", "F1", "MacroRecall", "CopperRecall"]
    all_out = os.path.join(args.result_dir, "PCI_INPUT_ABLATION_ALL_SEEDS.csv")
    summary_out = os.path.join(args.result_dir, "PCI_INPUT_ABLATION_MEAN_STD.csv")
    all_df.to_csv(all_out, index=False)

    summary = (
        all_df.groupby(["Delta", "Variant"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    for metric in metric_cols:
        summary[f"{metric}_percent_mean_std"] = summary.apply(
            lambda r, m=metric: f"{r[f'{m}_mean'] * 100:.2f} ± {r[f'{m}_std'] * 100:.2f}", axis=1
        )

    summary.to_csv(summary_out, index=False)
    display_cols = [
        "Delta", "Variant",
        "OA_percent_mean_std", "WGA_percent_mean_std", "F1_percent_mean_std",
        "MacroRecall_percent_mean_std", "CopperRecall_percent_mean_std",
    ]
    print("\n" + "=" * 130)
    print("PCI input ablation summary, mean ± std (%):")
    print("=" * 130)
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved all-seed results: {all_out}")
    print(f"Saved summary:          {summary_out}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--csv-path", type=str, default="/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    parser.add_argument("--weight-dir", type=str, default="weights")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--deltas", nargs="+", type=float, default=[-0.25, 0.0, 0.25])
    parser.add_argument("--m-ref", type=float, default=-0.5185)
    parser.add_argument("--dead-zone", type=float, default=0.08)
    parser.add_argument("--gain", type=float, default=1.1)
    parser.add_argument("--max-comp", type=float, default=0.25)
    parser.add_argument("--include-highlow-apc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    all_results = []
    for seed in args.seeds:
        all_results.append(evaluate_one_seed(args, seed))
    summarize(args, pd.concat(all_results, ignore_index=True))
