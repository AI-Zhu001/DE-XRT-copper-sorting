import argparse
from pathlib import Path

import pandas as pd


def main(args):
    result_dir = Path(args.result_dir)
    files = sorted(result_dir.glob("TRIPLE_SOTA_FULL_METRICS_TEST_CLIP_seed*.csv"))
    if not files:
        raise FileNotFoundError(f"No seed result CSV found in {result_dir}")

    dfs = []
    for f in files:
        print(f"Loading {f}")
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    metric_cols = ["OA", "WGA", "F1", "Recall"]
    summary = (
        df.groupby(["Delta", "Model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in c if str(x) != ""]).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]

    # paper-ready percent strings
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        summary[f"{metric}_percent_mean_std"] = summary.apply(
            lambda r: f"{r[mean_col] * 100:.1f} ± {r[std_col] * 100:.1f}", axis=1
        )

    raw_out = result_dir / "TRIPLE_SOTA_FULL_METRICS_TEST_CLIP_ALL_SEEDS.csv"
    summary_out = result_dir / "TRIPLE_SOTA_FULL_METRICS_TEST_CLIP_MEAN_STD.csv"
    df.to_csv(raw_out, index=False)
    summary.to_csv(summary_out, index=False)
    print(f"Saved raw combined: {raw_out}")
    print(f"Saved mean/std:     {summary_out}")

    # compact pivot for quick inspection
    compact_cols = ["OA_percent_mean_std", "WGA_percent_mean_std", "F1_percent_mean_std", "Recall_percent_mean_std"]
    print(summary[["Delta", "Model"] + compact_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, default="results")
    main(parser.parse_args())
