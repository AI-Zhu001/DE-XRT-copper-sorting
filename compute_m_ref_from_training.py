import os
import argparse
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms


def build_index(data_root: str):
    f_map = {}
    for p in Path(data_root).rglob("*"):
        if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
            name = p.name.replace(" ", "").lower()
            path_lower = str(p).lower()
            if "high" in path_lower:
                tag = "high"
            elif "low" in path_lower:
                tag = "low"
            else:
                continue
            f_map[(tag, name)] = str(p)
    print(f"Indexed image files: {len(f_map)}")
    return f_map


def sample_log_response(h: torch.Tensor, l: torch.Tensor, eps: float = 1e-6) -> float:
    # Eq.(6): m = 1/(2HW) * sum(log(H+eps) + log(L+eps))
    return 0.5 * (torch.log(h + eps) + torch.log(l + eps)).mean().item()


def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    f_map = build_index(args.data_root)
    df = pd.read_csv(args.csv_path)

    to_tensor = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    records = []
    missing = []

    for split in args.splits:
        split_df = df[df[args.split_col] == split].reset_index(drop=True)
        print(f"Computing log-response for split={split}, n={len(split_df)}")

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), leave=False):
            h_name = PureWindowsPath(row[args.high_col]).name.replace(" ", "").lower()
            l_name = PureWindowsPath(row[args.low_col]).name.replace(" ", "").lower()
            h_path = f_map.get(("high", h_name))
            l_path = f_map.get(("low", l_name))

            if h_path is None or l_path is None:
                missing.append({"split": split, "index": idx, "high_name": h_name, "low_name": l_name})
                continue

            h = to_tensor(Image.open(h_path).convert("L"))
            l = to_tensor(Image.open(l_path).convert("L"))
            m = sample_log_response(h, l, eps=args.eps)

            records.append({
                "split": split,
                "index": idx,
                "label": int(row[args.label_col]) if args.label_col in row else np.nan,
                "m": m,
            })

    out_samples = os.path.join(args.result_dir, "M_REF_SAMPLE_LOG_RESPONSES.csv")
    out_summary = os.path.join(args.result_dir, "M_REF_SPLIT_SUMMARY.csv")
    out_missing = os.path.join(args.result_dir, "M_REF_MISSING_PAIRS.csv")

    rec_df = pd.DataFrame(records)
    rec_df.to_csv(out_samples, index=False)

    summary = (
        rec_df.groupby("split")["m"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    summary.to_csv(out_summary, index=False)

    if missing:
        pd.DataFrame(missing).to_csv(out_missing, index=False)

    train_rows = summary[summary["split"] == "train"]
    if train_rows.empty:
        raise ValueError("No train split found. Check --split-col and split labels.")

    m_ref = float(train_rows["mean"].iloc[0])
    print("\n" + "=" * 90)
    print("Training-set global mean log-response m_ref computed using Eq.(6)")
    print("=" * 90)
    print(f"m_ref = {m_ref:.8f}")
    print("\nSplit summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved sample responses: {out_samples}")
    print(f"Saved split summary:    {out_summary}")
    if missing:
        print(f"Warning: missing image pairs: {len(missing)}; saved: {out_missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, default="/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集")
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--high-col", type=str, default="high_path")
    parser.add_argument("--low-col", type=str, default="low_path")
    parser.add_argument("--label-col", type=str, default="label")
    main(parser.parse_args())
