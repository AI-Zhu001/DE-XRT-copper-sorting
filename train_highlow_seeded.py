import os
import random
import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
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


class DEXRT_HighLowDataset(Dataset):
    def __init__(self, csv_path: str, f_map: dict, mode: str, image_size: int = 192):
        df = pd.read_csv(csv_path)
        self.samples = df[df["split"] == mode].reset_index(drop=True)
        self.f_map = f_map
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        h_name = PureWindowsPath(row["high_path"]).name.replace(" ", "").lower()
        l_name = PureWindowsPath(row["low_path"]).name.replace(" ", "").lower()
        h_path = self.f_map.get(("high", h_name))
        l_path = self.f_map.get(("low", l_name))
        if h_path is None or l_path is None:
            raise FileNotFoundError(f"Missing image pair: {h_name}, {l_name}")

        h = self.transform(Image.open(h_path).convert("L"))
        l = self.transform(Image.open(l_path).convert("L"))
        x = torch.cat([h, l], dim=0)
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, y


class TICL_HighLow_Net(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        old_conv = self.resnet.conv1
        new_conv = nn.Conv2d(
            2,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                old_w = old_conv.weight.data
                # Initialize both X-ray channels from the RGB-mean filter and rescale
                # to keep the initial activation magnitude close to the 3-channel model.
                new_w = old_w.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1) * (3.0 / 2.0)
                new_conv.weight.copy_(new_w)

        self.resnet.conv1 = new_conv
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
        feat = self.resnet(self.normalize(x))
        return self.classifier(feat)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().tolist())
            targets.extend(y.tolist())
    return accuracy_score(targets, preds), f1_score(targets, preds, average="macro")


def train_one_seed(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    f_map = build_index(args.data_root)

    train_ds = DEXRT_HighLowDataset(args.csv_path, f_map, "train", args.image_size)
    val_ds = DEXRT_HighLowDataset(args.csv_path, f_map, "val", args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TICL_HighLow_Net(pretrained=not args.no_pretrained).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.copper_weight], dtype=torch.float32, device=device))

    best_f1 = -1.0
    best_path = os.path.join(args.outdir, f"best_model_HIGHLOW_seed{args.seed}.pth")
    logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Seed {args.seed} | High+Low Epoch {epoch}/{args.epochs}")
        for x, y in loop:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        val_acc, val_f1 = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))
        logs.append({"Seed": args.seed, "Epoch": epoch, "TrainLoss": avg_loss, "ValAcc": val_acc, "ValF1": val_f1})
        print(f"Seed {args.seed} | Epoch {epoch} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"Saved best High+Low weight: {best_path}")

    log_path = os.path.join(args.logdir, f"highlow_train_seed{args.seed}.csv")
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print("=" * 80)
    print(f"Finished High+Low seed={args.seed}. Best Val F1={best_f1:.4f}")
    print(f"Best weight: {best_path}")
    print(f"Log: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-path", type=str, default="/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    parser.add_argument("--outdir", type=str, default="weights")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--copper-weight", type=float, default=1.5)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_one_seed(parse_args())
