# -*- coding: utf-8 -*-
"""Summary statistics reported in the paper, computed from the saved results.

- Tables 2 to 4: mean and SD over five seeds (results/results_final.csv)
- Welch's t tests between models on the five seed-level scores
- fitted fusion weights of the full model
- firm-level comparison used for Figure 8 (results/case/*_s0.json)
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

RES = "results"


def welch(df, a, b, metric):
    x, y = df[df.model == a][metric].values, df[df.model == b][metric].values
    return stats.ttest_ind(x, y, equal_var=False).pvalue


def main():
    df = pd.read_csv(f"{RES}/results_final.csv")
    summary = df.groupby("model")[["rmse", "mae", "r2", "auc", "f1", "acc"]].agg(["mean", "std"])
    print(summary.round(4).to_string(), "\n")

    print("Welch's t tests (two-sided, five seeds per model)")
    for b in ["mlp", "sage", "gcn", "gat"]:
        print(f"  RMSE  pecan_dual vs {b:5s}: p = {welch(df, 'pecan_dual', b, 'rmse'):.3f}")
    print(f"  AUC   pecan_dual_no_ff vs mlp: p = {welch(df, 'pecan_dual_no_ff', 'mlp', 'auc'):.3f}")
    print(f"  AUC   pecan_dual vs pecan_dual_no_ff: p = {welch(df, 'pecan_dual', 'pecan_dual_no_ff', 'auc'):.4f}")
    print(f"  AUC   pecan_dual vs pecan_dual_no_policy: p = {welch(df, 'pecan_dual', 'pecan_dual_no_policy', 'auc'):.3f}")
    print(f"  AUC   xgboost vs xgb_nopolicy: p = {welch(df, 'xgboost', 'xgb_nopolicy', 'auc'):.3f}")

    full = df[df.model == "pecan_dual"]
    print(f"\nFusion weight on the neural channel: regression {full.wr.mean():.2f} ± {full.wr.std():.2f}, "
          f"classification {full.wc.mean():.2f} ± {full.wc.std():.2f}")

    # Figure 8 context: firm-level comparison, seed 0
    load = lambda m: json.load(open(f"{RES}/case/{m}_s0.json"))
    p, mlp = load("pecan_dual"), load("mlp")
    y = np.array(p["y"])
    ep, em = np.abs(np.array(p["pred"]) - y), np.abs(np.array(mlp["pred"]) - y)
    print(f"\nSeed 0, {len(y)} test firms: PECAN intensity error < MLP error for {np.mean(ep < em):.1%} of firms")
    yc = np.array(p["yc"])
    right = (np.array(p["prob"]) > 0.5) == yc
    wrong_all = np.all([(np.array(load(m)["prob"]) > 0.5) != yc
                        for m in ["mlp", "gcn", "gat", "sage", "xgb_nopolicy"]], axis=0)
    print(f"Expansion: PECAN is the only model on the correct side of 0.5 for {int((right & wrong_all).sum())} firms")


if __name__ == "__main__":
    main()
