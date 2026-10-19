# -*- coding: utf-8 -*-
"""Case-study figure: individual forecasts where PECAN is right and the baselines are not.

Uses the seed-0 per-firm predictions dumped by train_case.py (results/case/*_s0.json),
firm text profiles (results/case/firm_text.jsonl) and the PECAN attention export.
"""
import json
import textwrap
from collections import Counter, defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = "."
RES, FIG = f"{BASE}/results", f"{BASE}/figures"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
C_PECAN, C_RIGHT, C_WRONG, C_GRAY, C_TEXT = "#C44E52", "#55A868", "#B5B5B5", "#8C8C8C", "#222222"
CARD_BG = ["#FDF3D9", "#E3EFF9", "#E6F2E3", "#F9E3DD"]

MODELS = [("pecan_dual", "PECAN"), ("mlp", "MLP"), ("gcn", "GCN"), ("gat", "GAT"),
          ("sage", "GraphSAGE"), ("xgb_nopolicy", "XGBoost")]

# English text for the four firms (translated from the Tonghuashun profiles) and the
# policy titles (translated from gov.cn); tools and binding strength come from the LLM output.
TOOLS_EN = {"财政补贴": "subsidy", "税收优惠": "tax incentive", "金融支持": "finance",
            "政府采购": "procurement", "人才支持": "talent", "技术研发支持": "R&D support",
            "基础设施建设": "infrastructure", "标准与规范": "standards", "市场准入": "market access",
            "试点示范": "pilot", "监管约束": "regulation"}
CASES = [
    dict(code="300436", title="Guangshengtang (300436) · Biopharma", task="both",
         biz="Develops, manufactures, and sells nucleoside anti-hepatitis B drugs; "
             "also liver, respiratory, and cardiovascular medicines.",
         policies=[("深化医药卫生体制改革2024年重点工作任务",
                    "Key tasks for deepening medical and health system reform (2024)"),
                   ("缓缴药品、医疗器械产品注册费",
                    "Deferral of drug and device registration fees (2022)")]),
    dict(code="300197", title="CECEP Techand (300197) · Environment", task="reg",
         biz="Ecological restoration and landscaping projects, environmental engineering, "
             "operation of pollution control facilities, and nursery stock.",
         policies=[("“十四五”节能减排综合工作方案",
                    "14th Five-Year comprehensive plan for energy saving and emission reduction (2022)"),
                   ("支持山东深化新旧动能转换",
                    "Supporting Shandong's green and low-carbon transition (2022)")]),
    dict(code="002129", title="TCL Zhonghuan (002129) · Electronic components", task="cls",
         biz="Monocrystalline silicon rods and wafers, solar cells and modules, and "
             "semiconductor materials and devices.",
         policies=[("高性能制造技术与重大装备",
                    "National key R&D programs incl. high-performance manufacturing (2024)")]),
    dict(code="600775", title="Nanjing Panda (600775) · Electronics & IT", task="cls",
         biz="Private wireless communication, advanced manufacturing services, intelligent "
             "equipment, and smart city systems.",
         policies=[("关于做好2024年中小企业数字化转型城市试点工作的通知",
                    "City pilots for the digital transformation of SMEs (2024)"),
                   ("真抓实干成效明显地方予以督查激励",
                    "Incentives for localities implementing major policies (2022)")]),
]


def load():
    preds = {m: json.load(open(f"{RES}/case/{m}_s0.json")) for m, _ in MODELS}
    att = json.load(open(f"{RES}/attention_dual_final.json"))
    meta = json.load(open(f"{BASE}/data/policy_meta.json"))
    fin = {json.loads(l)["code"]: json.loads(l)["fin"] for l in open(f"{BASE}/data/firm_fin.jsonl")}
    return preds, att, meta, fin


def policy_tags(att, meta, firm_idx, key):
    """LLM tools and binding strength of the linked policy whose title contains `key`."""
    for s, d in zip(att["src"], att["dst"]):
        if d != firm_idx:
            continue
        m = meta.get(att["policy_urls"][s], {})
        if key in m.get("title", ""):
            tools = ", ".join(TOOLS_EN.get(t, t) for t in (m.get("tools") or [])[:4])
            return f"LLM: {tools}; binding {m.get('strength')}/5"
    raise KeyError(key)


def wrap(s, w):
    return "\n".join(textwrap.wrap(s, w))


def reg_panel(ax, preds, i, y_true):
    labs = [lab for _, lab in MODELS][::-1]
    vals = [preds[m]["pred"][i] for m, _ in MODELS][::-1]
    lo, hi = min(vals + [y_true]), max(vals + [y_true])
    pad = 0.25 * (hi - lo)
    for k, (lab, v) in enumerate(zip(labs, vals)):
        col = C_PECAN if lab == "PECAN" else C_WRONG
        ax.plot([y_true, v], [k, k], color=col, lw=1.6 if lab == "PECAN" else 1.1, zorder=2,
                solid_capstyle="butt")
        ax.scatter([v], [k], s=22 if lab == "PECAN" else 14, color=col, zorder=3)
        ax.text(hi + 0.06 * (hi - lo + pad), k, f"{abs(v - y_true):.2f}", va="center", fontsize=5.8,
                color=C_PECAN if lab == "PECAN" else C_TEXT,
                fontweight="bold" if lab == "PECAN" else "normal")
    ax.axvline(y_true, color="black", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs)
    ax.set_xlim(lo - pad, hi + 0.32 * (hi - lo + pad))
    ax.set_xlabel("Forecast 2025 R&D intensity (%)")
    ax.text(0.0, 1.02, f"Observed: {y_true:.2f}%", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.3, fontweight="bold")
    ax.text(hi + 0.06 * (hi - lo + pad), len(labs) - 0.45, "error", fontsize=5.6, color=C_GRAY)


def cls_panel(ax, preds, i, yc):
    labs = [lab for _, lab in MODELS][::-1]
    probs = [preds[m]["prob"][i] for m, _ in MODELS][::-1]
    for k, (lab, p) in enumerate(zip(labs, probs)):
        right = (p > 0.5) == bool(yc)
        col = (C_PECAN if lab == "PECAN" else C_RIGHT) if right else C_WRONG
        ax.barh(k, p, color=col, height=0.62, zorder=2)
        ax.text(p + 0.015, k, f"{p:.2f}", va="center", fontsize=5.8, color=C_TEXT)
        ax.text(1.13, k, "\u2713" if right else "\u2717", va="center", ha="center", fontsize=6.5,
                color=C_RIGHT if right else "#9A3B3B", fontfamily="DejaVu Sans")
    ax.axvline(0.5, color="black", lw=0.6, ls=(0, (2, 2)), zorder=3)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("P(expand R&D in 2025)")
    ax.text(0.0, 1.02, "Observed: " + ("expanded" if yc else "did not expand"),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, fontweight="bold")


def main():
    preds, att, meta, fin = load()
    P = preds["pecan_dual"]
    pos = {c: i for i, c in enumerate(P["code"])}
    fig = plt.figure(figsize=(7.2, 7.6))
    W, H = 0.485, 0.48
    origins = [(0.0, 0.51), (0.515, 0.51), (0.0, 0.0), (0.515, 0.0)]
    LH = 0.0142  # one 6-pt text line in figure coordinates
    for k, (case, (ox, oy)) in enumerate(zip(CASES, origins)):
        i = pos[case["code"]]
        fidx = P["firm_idx"][i]
        fig.add_artist(FancyBboxPatch((ox + 0.005, oy + 0.005), W - 0.01, H - 0.01, transform=fig.transFigure,
                                      boxstyle="round,pad=0,rounding_size=0.018", facecolor=CARD_BG[k],
                                      edgecolor="none", zorder=-5))
        y = oy + H - 0.025
        fig.text(ox + 0.02, y, "ABCD"[k], fontsize=10, fontweight="bold", va="top")
        fig.text(ox + 0.055, y - 0.001, case["title"], fontsize=7.6, fontweight="bold", va="top")
        y -= 0.033
        biz = textwrap.wrap("Business: " + case["biz"], 70)
        fig.text(ox + 0.02, y, "\n".join(biz), fontsize=6.0, va="top", color=C_TEXT, linespacing=1.25)
        y -= LH * len(biz) + 0.008
        fin_c = fin[case["code"]]
        r24 = 100 * fin_c["2024"]["rd"] / fin_c["2024"]["revenue"]
        r25 = 100 * fin_c["2025"]["rd"] / fin_c["2025"]["revenue"]
        fig.text(ox + 0.02, y, f"R&D intensity: {r24:.2f}% in 2024, {r25:.2f}% in 2025", fontsize=6.0,
                 va="top", color=C_TEXT)
        y -= LH + 0.008
        fig.text(ox + 0.02, y, "Linked policies (LLM profile)", fontsize=6.3, fontweight="bold", va="top")
        y -= LH + 0.002
        for key, en in case["policies"]:
            lines = textwrap.wrap(en, 72)
            fig.text(ox + 0.025, y, "\u2022 " + "\n   ".join(lines), fontsize=5.9, va="top", color=C_TEXT,
                     linespacing=1.2)
            y -= LH * len(lines)
            fig.text(ox + 0.04, y, policy_tags(att, meta, fidx, key), fontsize=5.5, va="top", color=C_GRAY,
                     style="italic")
            y -= LH + 0.004
        top = y - 0.035  # leave room for the panel header text
        bottom = oy + 0.055
        hgt = top - bottom
        if case["task"] == "both":
            reg_panel(fig.add_axes([ox + 0.08, bottom, 0.15, hgt]), preds, i, P["y"][i])
            a2 = fig.add_axes([ox + 0.32, bottom, 0.12, hgt])
            cls_panel(a2, preds, i, P["yc"][i])
            a2.set_xlabel("P(expand)")
            [t for t in a2.texts if t.get_text().startswith("Observed")][0].set_text(
                "Did not expand" if not P["yc"][i] else "Expanded")
        elif case["task"] == "reg":
            reg_panel(fig.add_axes([ox + 0.1, bottom, 0.34, hgt]), preds, i, P["y"][i])
        else:
            cls_panel(fig.add_axes([ox + 0.1, bottom, 0.33, hgt]), preds, i, P["yc"][i])
    fig.savefig(f"{FIG}/fig_case_forecasts.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_case_forecasts.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
