# -*- coding: utf-8 -*-
"""Nature-style figures for the PECAN paper. Run locally after downloading results."""
import json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE = "."
RES, FIG = f"{BASE}/results", f"{BASE}/figures"
os.makedirs(FIG, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

C_MAIN = "#C44E52"   # PECAN
C_ABL = "#DD8452"
C_GNN = "#4C72B0"
C_TAB = "#55A868"
C_GRAY = "#8C8C8C"

LABELS = {
    "pecan_dual": "PECAN (ours)", "pecan": "PECAN (neural only)",
    "pecan_dual_no_llm": "w/o LLM struct.", "pecan_dual_no_policy": "w/o policy attention",
    "pecan_dual_no_semantic": "w/o semantic edges", "pecan_dual_no_ff": "w/o firm-firm GAT",
    "mlp": "MLP", "gcn": "GCN", "gat": "GAT", "sage": "GraphSAGE",
    "xgboost": "XGBoost", "xgb_nopolicy": "XGBoost (no policy)",
}

def load_results():
    df = pd.read_csv(f"{RES}/results_final.csv")
    return df

def fig_main(df):
    order = ["xgb_nopolicy", "xgboost", "gcn", "gat", "sage", "mlp", "pecan", "pecan_dual"]
    colors = [C_TAB, C_TAB, C_GNN, C_GNN, C_GNN, C_TAB, C_ABL, C_MAIN]
    metrics = [("rmse", "RMSE (pp)"), ("mae", "MAE (pp)"),
               ("r2", r"$R^2$"), ("auc", "AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.2))
    for ax, (m, lab) in zip(axes, metrics):
        for yi, mod in enumerate(order):
            sub = df[df.model == mod][m]
            ax.errorbar(sub.mean(), yi, xerr=sub.std(), fmt="o", ms=4.5,
                        color=colors[yi], ecolor=colors[yi], elinewidth=1.1, capsize=2.5,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS[o] for o in order] if ax is axes[0] else [""] * len(order))
        ax.set_xlabel(lab)
        ax.grid(axis="x", lw=0.35, color="#DDDDDD", zorder=0)
        best = df[df.model == "pecan_dual"][m].mean()
        ax.axvline(best, lw=0.6, ls="--", color=C_MAIN, alpha=0.55, zorder=1)
        ax.set_ylim(-0.6, len(order) - 0.4)
    for i, ax in enumerate(axes):
        ax.text(-0.02 if i else -0.62, 1.06, chr(65 + i), transform=ax.transAxes,
                fontsize=9.5, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{FIG}/fig_main_results.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_main_results.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

def fig_ablation(df):
    order = ["pecan_dual", "pecan_dual_no_llm", "pecan_dual_no_policy",
             "pecan_dual_no_semantic", "pecan_dual_no_ff", "pecan"]
    colors = [C_MAIN, C_ABL, C_ABL, C_ABL, C_ABL, C_ABL]
    metrics = [("rmse", "RMSE (pp)"), ("auc", "AUC"), ("f1", "F1")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1))
    for ax, (m, lab) in zip(axes, metrics):
        for yi, mod in enumerate(order):
            sub = df[df.model == mod][m]
            ax.errorbar(sub.mean(), yi, xerr=sub.std(), fmt="s" if mod != "pecan_dual" else "o",
                        ms=4.2, color=colors[yi], elinewidth=1.1, capsize=2.5,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS[o] for o in order] if ax is axes[0] else [""] * len(order))
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.grid(axis="x", lw=0.35, color="#DDDDDD", zorder=0)
    for i, ax in enumerate(axes):
        ax.text(-0.02 if i else -0.75, 1.07, chr(65 + i), transform=ax.transAxes,
                fontsize=9.5, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{FIG}/fig_ablation.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_ablation.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

def fig_scatter():
    d = json.load(open(f"{RES}/preds_dual_final.json"))
    y, p = np.array(d["y"]), np.array(d["pred"])
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4))
    ax = axes[0]
    ax.scatter(y, p, s=7, alpha=0.45, color=C_GNN, edgecolors="none", zorder=3)
    lim = [0, max(y.max(), p.max()) * 1.05]
    ax.plot(lim, lim, lw=0.8, ls="--", color=C_GRAY, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed R&D intensity in 2025 (%)")
    ax.set_ylabel("Predicted (%)")
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ax.text(0.05, 0.92, f"$R^2$ = {r2:.3f}\nn = {len(y)}", transform=ax.transAxes, fontsize=7)
    ax = axes[1]
    err = p - y
    ax.hist(err, bins=41, color=C_GNN, alpha=0.85, edgecolor="white", lw=0.3, zorder=3)
    ax.axvline(0, lw=0.7, ls="--", color=C_GRAY)
    ax.set_xlabel("Prediction error (pp)")
    ax.set_ylabel("Firms")
    for i, ax in enumerate(axes):
        ax.text(-0.18, 1.05, chr(65 + i), transform=ax.transAxes, fontsize=9.5, fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{FIG}/fig_scatter.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_scatter.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    df = load_results()
    fig_main(df)
    fig_ablation(df)
    fig_scatter()
    print("figures written to", FIG)
