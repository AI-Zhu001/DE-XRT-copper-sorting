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
# 1. Model definitions
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


class CBAL2_Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.stream_high = models.resnet18()
        self.stream_high.fc = nn.Identity()

        self.stream_low = models.resnet18()
        self.stream_low.fc = nn.Identity()

        self.attn_gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, h, l):
        h = h.repeat(1, 3, 1, 1)
        l = l.repeat(1, 3, 1, 1)

        f_h = self.stream_high(self.normalize(h))
        f_l = self.stream_low(self.normalize(l))

        f = torch.cat([f_h, f_l], dim=1)
        f = f * self.attn_gate(f)

        return self.classifier(f)


class MobileNetV3_DEXRT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v3_small()

        in_f = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_f, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.backbone(self.normalize(x))


class EfficientNetB0_DEXRT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.efficientnet_b0()
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.backbone(self.normalize(x))


# ============================================================
# 2. Soft APC
# ============================================================
def soft_apc_predict(model, h, l, device, m_ref=-0.5185):
    model.eval()
    eps = 1e-6

    with torch.no_grad():
        log_h = torch.log(h + eps)
        log_l = torch.log(l + eps)

        m_curr = 0.5 * (log_h + log_l).mean().item()
        diff = m_ref - m_curr
        abs_d = abs(diff)

        dead_zone = 0.08
        gain = 1.1
        max_comp = 0.25

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


# ============================================================
# 3. File index
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
# 4. Metrics
# ============================================================
def compute_group(row, h_raw, l_raw):
    eps = 1e-6
    m_orig = 0.5 * (
        torch.log(h_raw + eps) + torch.log(l_raw + eps)
    ).mean().item()

    if m_orig > -0.45:
        return "Thin"
    elif m_orig < -0.65:
        return "Thick"
    return "Medium"


def compute_metrics(y_true, y_pred, groups):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    oa = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    group_accs = []
    for g in np.unique(groups):
        mask = groups == g
        if np.any(mask):
            group_accs.append(accuracy_score(y_true[mask], y_pred[mask]))

    wga = min(group_accs) if group_accs else 0.0

    return oa, wga, f1, recall


# ============================================================
# 5. Evaluation
# ============================================================
def evaluate(seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    csv_path = "/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"
    data_root = "/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集"

    os.makedirs("results", exist_ok=True)

    # Weight paths
    w_proposed = f"weights/best_model_FINAL_seed{seed}.pth"
    w_cbal2 = f"weights/best_model_CBAL2_SOTA_seed{seed}.pth"
    w_mobile = f"weights/best_model_MOBILE_SOTA_seed{seed}.pth"
    w_eff = f"weights/best_model_EFFICIENTNET_B0_seed{seed}.pth"

    for path in [w_proposed, w_cbal2, w_mobile, w_eff]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing weight file: {path}")

    # Load models
    proposed_model = TICL_PCI_Net_Pro().to(device)
    proposed_model.load_state_dict(
        torch.load(w_proposed, map_location=device),
        strict=False
    )

    cbal2_model = CBAL2_Net().to(device)
    cbal2_model.load_state_dict(
        torch.load(w_cbal2, map_location=device),
        strict=True
    )

    mobile_model = MobileNetV3_DEXRT().to(device)
    mobile_model.load_state_dict(
        torch.load(w_mobile, map_location=device),
        strict=True
    )

    eff_model = EfficientNetB0_DEXRT().to(device)
    eff_model.load_state_dict(
        torch.load(w_eff, map_location=device),
        strict=True
    )

    proposed_model.eval()
    cbal2_model.eval()
    mobile_model.eval()
    eff_model.eval()

    f_map = build_index(data_root)

    df = pd.read_csv(csv_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor()
    ])

    shifts = [-0.25, -0.15, -0.05, 0, 0.05, 0.15, 0.25]
    records = []

    for delta in shifts:
        print(f"\nEvaluating delta={delta:+.2f}")

        preds = {
            "Baseline": [],
            "CBAL2": [],
            "MobileV3": [],
            "EfficientNet-B0": [],
            "Proposed": []
        }
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
            d_env = torch.log(h_env + eps) - torch.log(l_env + eps)
            x_env = torch.cat([h_env, l_env, d_env], dim=1)

            with torch.no_grad():
                # Baseline: same PCI model without Soft APC
                pred_base = proposed_model(x_env).argmax(1).item()

                # CBAL2: dual-stream H/L
                pred_cbal2 = cbal2_model(h_env, l_env).argmax(1).item()

                # MobileNetV3: PCI input
                pred_mobile = mobile_model(x_env).argmax(1).item()

                # EfficientNet-B0: PCI input
                pred_eff = eff_model(x_env).argmax(1).item()

            pred_prop = soft_apc_predict(proposed_model, h_env, l_env, device)

            preds["Baseline"].append(pred_base)
            preds["CBAL2"].append(pred_cbal2)
            preds["MobileV3"].append(pred_mobile)
            preds["EfficientNet-B0"].append(pred_eff)
            preds["Proposed"].append(pred_prop)

            targets.append(target)
            groups.append(compute_group(row, h_raw, l_raw))

        for model_name, y_pred in preds.items():
            oa, wga, f1, recall = compute_metrics(targets, y_pred, groups)

            records.append({
                "Delta": delta,
                "Model": model_name,
                "OA": oa,
                "WGA": wga,
                "F1": f1,
                "Recall": recall
            })

    out_path = f"results/FIVE_WAY_FULL_METRICS_TEST_CLIP_seed{seed}.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")

    report = pd.DataFrame(records)
    pivot = report.pivot(
        index="Delta",
        columns="Model",
        values=["OA", "WGA", "F1", "Recall"]
    )
    print(pivot)


# ============================================================
# 6. Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(args.seed)