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
    """
    Build a filename-based index for high- and low-energy images.
    The script identifies channel type from the full path containing 'high' or 'low'.
    """
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
    """
    Eq. (6):
        m = 1/(2HW) * sum(log(H + eps) + log(L + eps))

    h, l are expected to be grayscale tensors with shape [1, H, W],
    normalized to [0, 1] by transforms.ToTensor().
    """
    return 0.5 * (torch.log(h + eps) + torch.log(l + eps)).mean().item()


def safe_label(row, label_col: str, has_label: bool):
    if not has_label:
        return np.nan

    val = row[label_col]
    if pd.isna(val):
        return np.nan

    try:
        return int(val)
    except Exception:
        return val


def compute_split_summary(rec_df: pd.DataFrame, deltas):
    """
    Compute split-level statistics of mean log response m.
    std uses pandas default sample standard deviation, ddof=1.
    """
    if rec_df.empty:
        raise ValueError("No valid image pairs were processed. Check paths and split labels.")

    summary = (
        rec_df.groupby("split")["m"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            p1=lambda x: np.percentile(x, 1),
            p5=lambda x: np.percentile(x, 5),
            median="median",
            p95=lambda x: np.percentile(x, 95),
            p99=lambda x: np.percentile(x, 99),
            max="max",
        )
        .reset_index()
    )

    for delta in deltas:
        col = f"delta_{delta:.2f}_sigma"
        summary[col] = summary["std"].apply(
            lambda s: delta / s if pd.notna(s) and s > 0 else np.nan
        )

    return summary


def write_paper_context(summary: pd.DataFrame, result_dir: str, deltas):
    """
    Save a small text file containing the exact values needed for manuscript revision.
    """
    train_rows = summary[summary["split"] == "train"]
    if train_rows.empty:
        raise ValueError("No train split found. Check --split-col and split labels.")

    r = train_rows.iloc[0]

    out_path = os.path.join(result_dir, "M_REF_PAPER_CONTEXT.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Training-set mean log-response statistics for manuscript revision\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"N_train = {int(r['count'])}\n")
        f.write(f"mean = {r['mean']:.8f}\n")
        f.write(f"std = {r['std']:.8f}\n")
        f.write(f"p5 = {r['p5']:.8f}\n")
        f.write(f"p95 = {r['p95']:.8f}\n")
        f.write(f"min = {r['min']:.8f}\n")
        f.write(f"max = {r['max']:.8f}\n\n")

        for delta in deltas:
            col = f"delta_{delta:.2f}_sigma"
            f.write(f"delta = ±{delta:.2f} corresponds to approximately ±{r[col]:.2f} sigma_m\n")

        f.write("\nSuggested manuscript sentence:\n")
        f.write("-" * 80 + "\n")
        f.write(
            "To further contextualize the stress-test range, the training-set mean log response "
            f"had a mean of {r['mean']:.5f}, a standard deviation of "
            f"σ_m = {r['std']:.5f}, and a 5th–95th percentile range of "
            f"[{r['p5']:.5f}, {r['p95']:.5f}]. "
            f"Therefore, δ = ±0.25 corresponds to approximately "
            f"±{r['delta_0.25_sigma']:.2f}σ_m relative to the training response distribution, "
            "representing a severe controlled response-shift condition rather than a typical "
            "operating probability.\n"
        )

    return out_path


def main(args):
    os.makedirs(args.result_dir, exist_ok=True)

    f_map = build_index(args.data_root)
    df = pd.read_csv(args.csv_path)

    required_cols = [args.split_col, args.high_col, args.low_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    has_label = args.label_col in df.columns
    if not has_label:
        print(f"Warning: label column '{args.label_col}' not found. Labels will be saved as NaN.")

    to_tensor = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    records = []
    missing = []

    for split in args.splits:
        split_df = df[df[args.split_col] == split].reset_index(drop=True)
        print(f"\nComputing log-response for split={split}, n={len(split_df)}")

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), leave=False):
            try:
                h_name = PureWindowsPath(str(row[args.high_col])).name.replace(" ", "").lower()
                l_name = PureWindowsPath(str(row[args.low_col])).name.replace(" ", "").lower()
            except Exception as e:
                missing.append({
                    "split": split,
                    "index": idx,
                    "high_name": row.get(args.high_col, ""),
                    "low_name": row.get(args.low_col, ""),
                    "reason": f"path_parse_error: {e}",
                })
                continue

            h_path = f_map.get(("high", h_name))
            l_path = f_map.get(("low", l_name))

            if h_path is None or l_path is None:
                missing.append({
                    "split": split,
                    "index": idx,
                    "high_name": h_name,
                    "low_name": l_name,
                    "reason": "missing_high_or_low_image",
                })
                continue

            try:
                with Image.open(h_path) as img_h:
                    h = to_tensor(img_h.convert("L"))

                with Image.open(l_path) as img_l:
                    l = to_tensor(img_l.convert("L"))

                m = sample_log_response(h, l, eps=args.eps)

                records.append({
                    "split": split,
                    "index": idx,
                    "label": safe_label(row, args.label_col, has_label),
                    "high_name": h_name,
                    "low_name": l_name,
                    "m": m,
                })

            except Exception as e:
                missing.append({
                    "split": split,
                    "index": idx,
                    "high_name": h_name,
                    "low_name": l_name,
                    "reason": f"image_or_log_error: {e}",
                })
                continue

    rec_df = pd.DataFrame(records)

    out_samples = os.path.join(args.result_dir, "M_REF_SAMPLE_LOG_RESPONSES.csv")
    out_summary = os.path.join(args.result_dir, "M_REF_SPLIT_SUMMARY.csv")
    out_missing = os.path.join(args.result_dir, "M_REF_MISSING_PAIRS.csv")

    rec_df.to_csv(out_samples, index=False)

    summary = compute_split_summary(rec_df, args.deltas)
    summary.to_csv(out_summary, index=False)

    if missing:
        pd.DataFrame(missing).to_csv(out_missing, index=False)

    paper_context_path = write_paper_context(summary, args.result_dir, args.deltas)

    train_rows = summary[summary["split"] == "train"]
    if train_rows.empty:
        raise ValueError("No train split found. Check --split-col and split labels.")

    train = train_rows.iloc[0]
    m_ref = float(train["mean"])

    print("\n" + "=" * 90)
    print("Training-set global mean log-response m_ref computed using Eq. (6)")
    print("=" * 90)
    print(f"m_ref = {m_ref:.8f}")
    print(f"N_train = {int(train['count'])}")
    print(f"std = {train['std']:.8f}")
    print(f"p5 = {train['p5']:.8f}")
    print(f"p95 = {train['p95']:.8f}")
    print(f"delta ±0.25 = ±{train['delta_0.25_sigma']:.2f} sigma_m")

    print("\nSplit summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved sample responses: {out_samples}")
    print(f"Saved split summary:    {out_summary}")
    print(f"Saved paper context:    {paper_context_path}")

    if missing:
        print(f"Warning: missing/problematic image pairs: {len(missing)}; saved: {out_missing}")
    else:
        print("Missing/problematic image pairs: 0")

    # Consistency checks. These are warnings, not hard stops.
    if args.expected_train_count is not None:
        if int(train["count"]) != args.expected_train_count:
            print(
                f"\nWarning: train count is {int(train['count'])}, "
                f"but expected {args.expected_train_count}. Check split file and missing pairs."
            )

    if args.expected_m_ref is not None:
        diff = abs(m_ref - args.expected_m_ref)
        if diff > args.m_ref_tolerance:
            print(
                f"\nWarning: computed m_ref differs from expected value. "
                f"computed={m_ref:.8f}, expected={args.expected_m_ref:.8f}, diff={diff:.8f}. "
                "Do not copy results into the manuscript until this is explained."
            )
        else:
            print(
                f"\nCheck passed: computed m_ref is close to expected value "
                f"{args.expected_m_ref:.8f}."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv-path",
        type=str,
        default="/root/projects/Hou_swin/split_outputs/copper_xray_all_splits.csv",
        help="CSV file containing split, high_path, low_path, and label columns.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/root/autodl-tmp/data/原始购买的二分类数据集/原始购买的二分类数据集",
        help="Root directory containing high- and low-energy images.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="Directory to save output CSV and text files.",
    )

    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--eps", type=float, default=1e-6)

    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--high-col", type=str, default="high_path")
    parser.add_argument("--low-col", type=str, default="low_path")
    parser.add_argument("--label-col", type=str, default="label")

    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=[0.05, 0.15, 0.25],
        help="Offset values to express as multiples of training-set sigma_m.",
    )

    parser.add_argument(
        "--expected-train-count",
        type=int,
        default=5071,
        help="Expected number of valid training pairs. Use None by passing --expected-train-count -1.",
    )
    parser.add_argument(
        "--expected-m-ref",
        type=float,
        default=-0.51847,
        help="Expected m_ref reported in the manuscript.",
    )
    parser.add_argument(
        "--m-ref-tolerance",
        type=float,
        default=5e-4,
        help="Tolerance for comparing computed m_ref with expected m_ref.",
    )

    args = parser.parse_args()

    if args.expected_train_count == -1:
        args.expected_train_count = None

    main(args)
