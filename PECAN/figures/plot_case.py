# -*- coding: utf-8 -*-
"""Case study figure: policy-firm coupling from PECAN attention (v2)."""
import json
import numpy as np
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE = "."
RES, FIG = f"{BASE}/results", f"{BASE}/figures"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})
C_MAIN, C_GNN, C_GRAY = "#C44E52", "#4C72B0", "#8C8C8C"

IND_EN = {"电子信息": "Electronics & IT", "电子器件": "Electronic comp.", "机械行业": "Machinery",
          "汽车制造": "Automobile", "生物制药": "Biopharma", "医疗器械": "Medical devices",
          "电器行业": "Electrical appl.", "发电设备": "Power equipment", "飞机制造": "Aircraft",
          "仪器仪表": "Instruments", "环保行业": "Environment", "化工行业": "Chemicals",
          "船舶制造": "Shipbuilding"}

POLICY_KEY_EN = [
    ("稀土管理条例", "Rare-Earth Administration Regulation (2024)"),
    ("原材料工业数字化转型", "Raw-Material Industry Digitalization Plan (2024)"),
    ("2024—2025年节能降碳行动方案", "Energy Saving & Carbon Reduction Action (2024)"),
    ("高性能制造技术与重大装备", "Key R&D Program: High-Perf. Manufacturing (2024)"),
    ("应急机器人", "Emergency Robotics Guideline (2024)"),
    ("充电基础设施", "EV Charging Infra. & NEV Rural Rollout (2023)"),
    ("城市燃气管道", "Urban Gas Pipeline Renewal Program (2022)"),
    ("石化行业智能制造", "Petrochemical Smart Manufacturing Stds. (2022)"),
    ("“十四五”节能减排", "14th FYP Energy Saving & Emission Plan (2022)"),
    ("现代综合交通运输体系", "Modern Comprehensive Transport Plan (2022)"),
    ("非粮生物基材料", "Bio-based Materials Innovation Action (2023)"),
    ("国际邮轮", "Intl. Cruise Port Supply Regulation (2023)"),
    ("专精特新中小企业就业创业", "SRDI SME Employment Sailing Program (2023)"),
    ("河套深港科技创新", "Hetao SZ-HK S&T Cooperation Zone Plan (2023)"),
]

FIRM_EN = {"有研新材": "GRINM Adv. Materials", "中船防务": "CSSC Offshore & Marine",
           "航发动力": "AECC Aviation Power", "紫竹高科": "Zizhu High-Tech",
           "海特高新": "Haite High-Tech", "汇川技术": "Inovance",
           "聚光科技": "Focused Photonics", "雪迪龙": "SDL Technology",
           "林洋能源": "Linyang Energy", "大华股份": "Dahua Technology",
           "金卡智能": "Goldcard Smart", "汇中股份": "Sonic Holdings"}

def policy_en(title):
    for k, v in POLICY_KEY_EN:
        if k in title:
            return v
    return None

def main():
    att = json.load(open(f"{RES}/attention_dual_final.json"))
    meta = json.load(open(f"{BASE}/data/policy_meta.json"))
    src, dst, al = att["src"], att["dst"], att["alpha"]
    urls, names, inds = att["policy_urls"], att["names"], att["industry"]
    nfe = defaultdict(int)
    for d in dst:
        nfe[d] += 1

    # per-policy mean alpha over firms with >=5 edges
    agg = defaultdict(list)
    agg_ind = defaultdict(lambda: defaultdict(list))
    for k in range(len(al)):
        if nfe[dst[k]] >= 5:
            agg[src[k]].append(al[k])
            agg_ind[src[k]][inds[dst[k]]].append(al[k])
    rows = []
    seen = set()
    for s, v in agg.items():
        if len(v) < 30:
            continue
        lab = policy_en(meta.get(urls[s], {}).get("title", ""))
        if lab is None or lab in seen:
            continue
        seen.add(lab)
        rows.append((np.mean(v), len(v), s, lab))
    rows.sort(key=lambda r: -r[0])
    top8 = rows[:8]
    top6 = rows[:6]

    # curated selective pairs
    pairs = []
    for k in range(len(al)):
        if nfe[dst[k]] < 5:
            continue
        fn = names[dst[k]]
        if fn not in FIRM_EN:
            continue
        lab = policy_en(meta.get(urls[src[k]], {}).get("title", ""))
        if lab is None:
            continue
        pairs.append((al[k], FIRM_EN[fn], IND_EN.get(inds[dst[k]], ""), lab))
    pairs.sort(key=lambda x: -x[0])
    dedup, seenp = [], set()
    for a, f, i, p in pairs:
        if (f, p) in seenp:
            continue
        seenp.add((f, p))
        dedup.append((a, f, i, p))
    sel = dedup[:9]

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1],
                          hspace=0.62, wspace=0.66)

    # (a) policies by mean coupling strength
    ax_a = fig.add_subplot(gs[0, 0])
    labs = [r[3] for r in top8][::-1]
    vals = [r[0] for r in top8][::-1]
    ns = [r[1] for r in top8][::-1]
    ax_a.hlines(range(len(vals)), 0, vals, color=C_GNN, lw=1.4, zorder=2)
    ax_a.plot(vals, range(len(vals)), "o", ms=5, color=C_GNN,
              markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    for yi, (v, n) in enumerate(zip(vals, ns)):
        ax_a.text(v + 0.002, yi, f"n={n}", va="center", fontsize=5.6, color=C_GRAY)
    ax_a.set_yticks(range(len(vals)))
    ax_a.set_yticklabels(labs, fontsize=6.0)
    ax_a.set_xlabel(r"Mean coupling strength $\bar{\alpha}$")
    ax_a.set_xlim(0, max(vals) * 1.22)
    ax_a.grid(axis="x", lw=0.35, color="#E3E3E3", zorder=0)

    # (b) dot matrix top-6 policies x industries
    ax_b = fig.add_subplot(gs[0, 1])
    ind_list = list(IND_EN.keys())
    for xi, (mv, n, s, lab) in enumerate(top6):
        for yi, ind in enumerate(ind_list):
            v = agg_ind[s].get(ind, [])
            if len(v) >= 2:
                ax_b.scatter(xi, yi, s=6 + 3500 * np.mean(v), color=C_MAIN, alpha=0.75,
                             edgecolors="white", linewidths=0.4, zorder=3)
    ax_b.set_xticks(range(len(top6)))
    ax_b.set_xticklabels([f"P{k+1}" for k in range(len(top6))], fontsize=6.5)
    ax_b.set_yticks(range(len(ind_list)))
    ax_b.set_yticklabels([IND_EN[i] for i in ind_list], fontsize=6)
    ax_b.set_xlabel("Policy (P1–P6, panel A top-6)")
    ax_b.invert_yaxis()
    ax_b.grid(lw=0.3, color="#EEEEEE", zorder=0)
    for a_val, a_lab in [(0.02, "0.02"), (0.05, "0.05"), (0.10, "0.10")]:
        ax_b.scatter([], [], s=6 + 3500 * a_val, color=C_MAIN, alpha=0.75, label=a_lab)
    ax_b.legend(title=r"mean $\alpha$", loc="center left", bbox_to_anchor=(1.04, 0.5),
                frameon=False, labelspacing=1.0, title_fontsize=6.5, fontsize=6.2)

    # (c) selective policy-firm couplings
    ax_c = fig.add_subplot(gs[1, 0])
    sel_r = sel[::-1]
    ax_c.hlines(range(len(sel_r)), 0, [x[0] for x in sel_r], color=C_MAIN, lw=1.4, zorder=2)
    ax_c.plot([x[0] for x in sel_r], range(len(sel_r)), "o", ms=5, color=C_MAIN,
              markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax_c.set_yticks(range(len(sel_r)))
    ax_c.set_yticklabels([f"{f}  $\\leftarrow$  {p.split(' (')[0]}" for a, f, i, p in sel_r],
                         fontsize=6.0)
    ax_c.set_xlabel(r"Coupling strength $\alpha$")
    ax_c.grid(axis="x", lw=0.35, color="#E3E3E3", zorder=0)

    # (d) mean alpha by edge channel with bootstrap CI
    ax_d = fig.add_subplot(gs[1, 1])
    match_a, sem_a = [], []
    for k in range(len(al)):
        tinds = meta.get(urls[src[k]], {}).get("industries") or []
        (match_a if inds[dst[k]] in tinds else sem_a).append(al[k])
    rng = np.random.default_rng(0)
    def ci(x):
        x = np.array(x)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
        return np.percentile(bs, [2.5, 97.5])
    means = [np.mean(match_a), np.mean(sem_a)]
    cis = [ci(match_a), ci(sem_a)]
    for xi, (mn, (lo, hi), col, lab) in enumerate(zip(means, cis, [C_MAIN, C_GNN],
                  [f"Industry-targeted\n(n={len(match_a):,})", f"Semantic-only\n(n={len(sem_a):,})"])):
        ax_d.bar(xi, mn, width=0.55, color=col, alpha=0.85, zorder=3)
        ax_d.errorbar(xi, mn, yerr=[[mn - lo], [hi - mn]], fmt="none", ecolor="black",
                      elinewidth=0.9, capsize=3, zorder=4)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels([f"Industry-targeted\nedges (n={len(match_a):,})",
                          f"Semantic-only\nedges (n={len(sem_a):,})"], fontsize=6.3)
    ax_d.set_ylabel(r"Mean coupling strength $\bar{\alpha}$")
    rel = 100 * (means[0] / means[1] - 1)
    ax_d.text(0.5, 0.9, f"+{rel:.0f}%", transform=ax_d.transAxes, ha="center",
              fontsize=8, color=C_MAIN, fontweight="bold")

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "ABCD"):
        ax.text(-0.02, 1.1, letter, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", ha="right")
    fig.savefig(f"{FIG}/fig_case_study.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_case_study.png", dpi=400, bbox_inches="tight")
    print("P1-P6:", [(f"P{i+1}", r[3]) for i, r in enumerate(top6)])
    print("pairs:", [(round(a, 3), f, p) for a, f, i, p in sel])

if __name__ == "__main__":
    main()
