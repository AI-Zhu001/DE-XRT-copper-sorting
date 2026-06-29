import os
import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torchvision import transforms, models


class TICL_PCI_Net_Pro(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        return self.classifier(self.resnet(self.normalize(x)))


class CBAL2_Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stream_high = models.resnet18()
        self.stream_high.fc = nn.Identity()
        self.stream_low = models.resnet18()
        self.stream_low.fc = nn.Identity()
        self.attn_gate = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 1024), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, h, l):
        f_h = self.stream_high(self.normalize(h.repeat(1, 3, 1, 1)))
        f_l = self.stream_low(self.normalize(l.repeat(1, 3, 1, 1)))
        f_c = torch.cat([f_h, f_l], dim=1)
        return self.classifier(f_c * self.attn_gate(f_c))


class MobileNetV3_DEXRT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v3_small()
        in_f = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_f, 512), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        return self.backbone(self.normalize(x))


def build_index(data_root: str):
    f_map = {}
    for p in Path(data_root).rglob("*"):
        if p.suffix.lower() in [".jpg", ".png"]:
            key = p.name.replace(" ", "").lower()
            tag = "high" if "high" in str(p).lower() else "low"
            f_map[(tag, key)] = str(p)
    return f_map


def soft_apc_predict(model, h, l, m_ref=-0.5185):
    eps = 1e-6
    with torch.no_grad():
        log_h = torch.log(h + eps)
        log_l = torch.log(l + eps)
        m_curr = 0.5 * (log_h + log_l).mean().item()
        diff = m_ref - m_curr
        abs_d = abs(diff)
        dead_zone, gain, max_comp = 0.08, 1.1, 0.25
        if abs_d > dead_zone:
            d_comp = np.sign(diff) * (abs_d - dead_zone) * gain
        else:
            d_comp = 0.0
        d_comp = float(np.clip(d_comp, -max_comp, max_comp))
        h_c = torch.clamp(torch.exp(log_h + d_comp), 0, 1)
        l_c = torch.clamp(torch.exp(log_l + d_comp), 0, 1)
        d_c = torch.log(h_c + eps) - torch.log(l_c + eps)
        x = torch.cat([h_c, l_c, d_c], dim=1)
        return model(x).argmax(1).item()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.result_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Evaluating seed: {args.seed}")

    proposed_weight = os.path.join(args.weight_dir, f"best_model_FINAL_seed{args.seed}.pth")
    cbal2_weight = os.path.join(args.weight_dir, f"best_model_CBAL2_SOTA_seed{args.seed}.pth")
    mobile_weight = os.path.join(args.weight_dir, f"best_model_MOBILE_SOTA_seed{args.seed}.pth")

    for path in [proposed_weight, cbal2_weight, mobile_weight]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing weight file: {path}")

    p_model = TICL_PCI_Net_Pro().to(device)
    p_model.load_state_dict(torch.load(proposed_weight, map_location=device), strict=False)
    c_model = CBAL2_Net().to(device)
    c_model.load_state_dict(torch.load(cbal2_weight, map_location=device))
    m_model = MobileNetV3_DEXRT().to(device)
    m_model.load_state_dict(torch.load(mobile_weight, map_location=device))
    p_model.eval(); c_model.eval(); m_model.eval()

    f_map = build_index(args.data_root)
    df = pd.read_csv(args.csv_path)
    test_samples = df[df["split"] == "test"].copy().reset_index(drop=True)
    print(f"Test samples: {len(test_samples)}")
    to_tensor = transforms.Compose([transforms.Resize((192, 192)), transforms.ToTensor()])

    shifts = [-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25]
    final_stats = []

    for delta in shifts:
        print(f"Evaluating delta={delta:+.2f}")
        res = {"Proposed": [], "CBAL2": [], "MobileV3": [], "Baseline": [], "target": [], "group": []}
        for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), leave=False):
            hk = PureWindowsPath(row["high_path"]).name.replace(" ", "").lower()
            lk = PureWindowsPath(row["low_path"]).name.replace(" ", "").lower()
            hp = f_map.get(("high", hk))
            lp = f_map.get(("low", lk))
            if not hp or not lp:
                continue

            h_raw = to_tensor(Image.open(hp).convert("L")).unsqueeze(0).to(device)
            l_raw = to_tensor(Image.open(lp).convert("L")).unsqueeze(0).to(device)
            target = int(row["label"])

            h_env = torch.clamp(torch.exp(torch.log(h_raw + 1e-6) + delta), 0, 1)
            l_env = torch.clamp(torch.exp(torch.log(l_raw + 1e-6) + delta), 0, 1)
            diff_env = torch.log(h_env + 1e-6) - torch.log(l_env + 1e-6)
            x_env = torch.cat([h_env, l_env, diff_env], dim=1)

            res["Proposed"].append(soft_apc_predict(p_model, h_env, l_env))
            with torch.no_grad():
                res["CBAL2"].append(c_model(h_env, l_env).argmax(1).item())
                res["MobileV3"].append(m_model(x_env).argmax(1).item())
                res["Baseline"].append(p_model(x_env).argmax(1).item())

            m_orig = 0.5 * (torch.log(h_raw + 1e-6) + torch.log(l_raw + 1e-6)).mean().item()
            group = "Thin" if m_orig > -0.45 else ("Thick" if m_orig < -0.65 else "Medium")
            res["target"].append(target)
            res["group"].append(group)

        res_df = pd.DataFrame(res)
        for model_name in ["Proposed", "CBAL2", "MobileV3", "Baseline"]:
            y_t = res_df["target"]
            y_p = res_df[model_name]
            oa = accuracy_score(y_t, y_p)
            f1 = f1_score(y_t, y_p, average="macro")
            recall = recall_score(y_t, y_p, average="macro")
            g_accs = []
            for g in res_df["group"].unique():
                g_df = res_df[res_df["group"] == g]
                g_accs.append(accuracy_score(g_df["target"], g_df[model_name]))
            wga = min(g_accs) if g_accs else 0.0
            final_stats.append({
                "Seed": args.seed,
                "Delta": delta,
                "Model": model_name,
                "OA": oa,
                "WGA": wga,
                "F1": f1,
                "Recall": recall,
            })

    report = pd.DataFrame(final_stats)
    out_path = os.path.join(args.result_dir, f"TRIPLE_SOTA_FULL_METRICS_TEST_CLIP_seed{args.seed}.csv")
    report.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(report.pivot(index="Delta", columns="Model", values=["OA", "WGA", "F1", "Recall"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-path", type=str, default="/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    parser.add_argument("--weight-dir", type=str, default="weights")
    parser.add_argument("--result-dir", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
