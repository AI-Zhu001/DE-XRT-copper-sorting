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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, accuracy_score


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
# 1. File index
# ============================================================
def build_index(data_root):
    f_map = {}
    root = Path(data_root)

    for p in root.rglob("*"):
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
# 2. Dataset: PCI input [H, L, log(H)-log(L)]
# ============================================================
class DEXRT_EfficientNet_Dataset(Dataset):
    def __init__(self, csv_path, f_map, mode="train"):
        df = pd.read_csv(csv_path)
        self.samples = df[df["split"] == mode].reset_index(drop=True)
        self.f_map = f_map

        self.transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def _get_path(self, raw_path, tag):
        name = PureWindowsPath(raw_path).name.replace(" ", "").lower()
        path = self.f_map.get((tag, name))

        if path is None:
            raise FileNotFoundError(f"Missing {tag} image: {name}")

        return path

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        h_path = self._get_path(row["high_path"], "high")
        l_path = self._get_path(row["low_path"], "low")

        h = self.transform(Image.open(h_path).convert("L"))
        l = self.transform(Image.open(l_path).convert("L"))

        eps = 1e-6
        d = torch.log(h + eps) - torch.log(l + eps)

        x = torch.cat([h, l, d], dim=0)
        y = torch.tensor(int(row["label"]), dtype=torch.long)

        return x, y


# ============================================================
# 3. EfficientNet-B0 model
# ============================================================
class EfficientNetB0_DEXRT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

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
        x = self.normalize(x)
        return self.backbone(x)


# ============================================================
# 4. Validation
# ============================================================
def evaluate(model, loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            preds.extend(pred.cpu().tolist())
            targets.extend(y.cpu().tolist())

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")

    return acc, macro_f1


# ============================================================
# 5. Training
# ============================================================
def train(seed):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    csv_path = "/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv"
    data_root = "/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集"

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    f_map = build_index(data_root)

    train_ds = DEXRT_EfficientNet_Dataset(csv_path, f_map, mode="train")
    val_ds = DEXRT_EfficientNet_Dataset(csv_path, f_map, mode="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = EfficientNetB0_DEXRT(num_classes=2).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-3
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 1.5], dtype=torch.float32).to(device)
    )

    best_f1 = 0.0
    best_path = f"weights/best_model_EFFICIENTNET_B0_seed{seed}.pth"

    log_records = []

    for epoch in range(1, 21):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(
            train_loader,
            desc=f"Seed {seed} | EfficientNet-B0 Epoch {epoch}/20"
        )

        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / len(train_loader)

        print(
            f"Seed {seed} | Epoch {epoch} | "
            f"Loss {avg_loss:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}"
        )

        log_records.append({
            "Seed": seed,
            "Epoch": epoch,
            "TrainLoss": avg_loss,
            "ValAcc": val_acc,
            "ValF1": val_f1
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"Saved best EfficientNet-B0 weight: {best_path}")

    log_df = pd.DataFrame(log_records)
    log_df.to_csv(f"logs/efficientnet_b0_train_seed{seed}.csv", index=False)

    print("=" * 80)
    print(f"EfficientNet-B0 training finished. Seed={seed}")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Best weight: {best_path}")
    print("=" * 80)


# ============================================================
# 6. Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args.seed)