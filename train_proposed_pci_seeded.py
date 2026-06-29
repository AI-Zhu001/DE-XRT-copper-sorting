import os
import random
import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


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
        if p.suffix.lower() in [".jpg", ".png"]:
            tag = "high" if "high" in str(p).lower() else "low"
            key = p.name.replace(" ", "").lower()
            f_map[(tag, key)] = str(p)
    return f_map


class DEXRT_PCIDataset(Dataset):
    def __init__(self, csv_path: str, f_map: dict, mode: str):
        df = pd.read_csv(csv_path)
        self.samples = df[df["split"] == mode].reset_index(drop=True)
        self.f_map = f_map
        self.transform = transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        h_name = PureWindowsPath(row["high_path"]).name.replace(" ", "").lower()
        l_name = PureWindowsPath(row["low_path"]).name.replace(" ", "").lower()
        hp = self.f_map.get(("high", h_name))
        lp = self.f_map.get(("low", l_name))
        if not hp or not lp:
            raise KeyError(f"Missing image pair: {h_name}, {l_name}")

        h = self.transform(Image.open(hp).convert("L"))
        l = self.transform(Image.open(lp).convert("L"))
        d = torch.log(h + 1e-6) - torch.log(l + 1e-6)
        x = torch.cat([h, l, d], dim=0)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


class TICL_PCI_Net_Pro(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
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
        feat = self.resnet(self.normalize(x))
        return self.classifier(feat)


def train_one_seed(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("Building image index...")
    f_map = build_index(args.data_root)

    train_ds = DEXRT_PCIDataset(args.csv_path, f_map, "train")
    val_ds = DEXRT_PCIDataset(args.csv_path, f_map, "val")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = TICL_PCI_Net_Pro().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], device=device))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_path = os.path.join(args.outdir, f"best_model_FINAL_seed{args.seed}.pth")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Seed {args.seed} | Proposed Epoch {epoch + 1}/{args.epochs}")
        for x, y in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device, non_blocking=True))
                preds.extend(logits.argmax(1).cpu().tolist())
                targets.extend(y.tolist())
        f1 = f1_score(targets, preds, average="macro")
        acc = accuracy_score(targets, preds)
        print(f"Seed {args.seed} | Epoch {epoch + 1} | Loss {total_loss / len(train_loader):.4f} | Val Acc {acc:.4f} | Val F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            print(f"Saved best Proposed weight: {best_path}")

    print(f"Finished Proposed seed {args.seed}. Best Val F1: {best_f1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-path", type=str, default="/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    parser.add_argument("--outdir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    train_one_seed(parse_args())
