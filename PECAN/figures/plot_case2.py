# -*- coding: utf-8 -*-
"""Second case study figure: bipartite coupling network + industry scatter + firm vignette."""
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
C_MAIN, C_GNN, C_GRAY, C_GREEN = "#C44E52", "#4C72B0", "#8C8C8C", "#55A868"
IND_COLORS = {"电子器件": "#4C72B0", "船舶制造": "#55A868", "飞机制造": "#C44E52",
              "仪器仪表": "#8172B2", "电子信息": "#CCB974", "机械行业": "#64B5CD",
              "医疗器械": "#DD8452"}
IND_EN = {"电子信息": "Electronics & IT", "电子器件": "Electronic comp.", "机械行业": "Machinery",
          "汽车制造": "Automobile", "生物制药": "Biopharma", "医疗器械": "Medical devices",
          "电器行业": "Electrical appl.", "发电设备": "Power equipment", "飞机制造": "Aircraft",
          "仪器仪表": "Instruments", "环保行业": "Environment", "化工行业": "Chemicals",
          "船舶制造": "Shipbuilding"}
POLICY_KEY_EN = [
    ("稀土管理条例", "Rare-Earth Admin.\nRegulation (2024)"),
    ("现代综合交通运输体系", "Modern Comprehensive\nTransport Plan (2022)"),
    ("非粮生物基材料", "Bio-based Materials\nInnovation Action (2023)"),
    ("应急机器人", "Emergency Robotics\nGuideline (2024)"),
    ("专精特新中小企业就业创业", "SRDI SME Employment\nProgram (2023)"),
    ("河套深港科技创新", "Hetao SZ-HK S&T\nZone Plan (2023)"),
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
    fin = {json.loads(l)["code"]: json.loads(l)["fin"] for l in open(f"{BASE}/data/firm_fin.jsonl")}
    src, dst, al = att["src"], att["dst"], att["alpha"]
    urls, codes, inds, names = att["policy_urls"], att["codes"], att["industry"], att["names"]
    nfe = defaultdict(int)
    for d in dst:
        nfe[d] += 1

    # --- panel a data: bipartite edges among mapped policies/firms
    edges = []
    for k in range(len(al)):
        if nfe[dst[k]] < 5:
            continue
        pl = policy_en(meta.get(urls[src[k]], {}).get("title", ""))
        fn = names[dst[k]]
        if pl is None or fn not in FIRM_EN:
            continue
        edges.append((pl, FIRM_EN[fn], inds[dst[k]], al[k]))
    # dedup keep max alpha
    best = {}
    for p, f, i, a in edges:
        if (p, f) not in best or a > best[(p, f)][3]:
            best[(p, f)] = (p, f, i, a)
    edges = [v for v in best.values() if v[3] >= 0.05]
    pols = sorted({e[0] for e in edges}, key=lambda p: -sum(e[3] for e in edges if e[0] == p))
    firms = sorted({(e[1], e[2]) for e in edges}, key=lambda f: -max(e[3] for e in edges if e[1] == f[0]))
    p_pos = {p: 1 - i / max(len(pols) - 1, 1) for i, p in enumerate(pols)}
    f_pos = {f[0]: 1 - i / max(len(firms) - 1, 1) for i, f in enumerate(firms)}
    f_ind = dict(firms)

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.25, 1],
                          hspace=0.5, wspace=0.35)
    ax_a = fig.add_subplot(gs[:, 0])
    for p, f, i, a in edges:
        ax_a.plot([0.06, 0.94], [p_pos[p], f_pos[f]], "-", lw=0.6 + 24 * a,
                  color=C_GRAY, alpha=0.28 + 2.4 * a if a < 0.3 else 0.95, zorder=1)
    for p, y in p_pos.items():
        ax_a.plot(0.06, y, "s", ms=9, color=C_GNN, markeredgecolor="white", zorder=3)
        ax_a.text(0.03, y, p, ha="right", va="center", fontsize=6.0)
    for f, y in f_pos.items():
        col = IND_COLORS.get(f_ind[f], C_MAIN)
        ax_a.plot(0.94, y, "o", ms=8, color=col, markeredgecolor="white", zorder=3)
        ax_a.text(0.97, y, f"{f}\n({IND_EN.get(f_ind[f], '')})", ha="left", va="center", fontsize=6.0)
    ax_a.set_xlim(-0.55, 1.62)
    ax_a.set_ylim(-0.08, 1.08)
    ax_a.axis("off")
    ax_a.set_title("Policy-firm coupling network (edge width scales with α)", fontsize=7.5)
    for a_val in [0.05, 0.1, 0.2]:
        ax_a.plot([], [], "-", lw=0.6 + 24 * a_val, color=C_GRAY, label=f"α={a_val}")
    ax_a.legend(loc="lower center", ncol=3, frameon=False, fontsize=6.2,
                bbox_to_anchor=(0.5, -0.06))

    # --- panel b: industry mean coupling vs mean R&D intensity 2024
    def ri(x):
        if not x or not x.get("revenue") or x["revenue"] <= 0:
            return None
        rd = x.get("rd") or 0
        rd = 0 if rd != rd else rd
        return 100 * rd / x["revenue"]
    firm_coup = defaultdict(float)
    for k in range(len(al)):
        rel = meta.get(urls[src[k]], {}).get("relevance") or 0
        if rel >= 0.7:
            firm_coup[dst[k]] += al[k]
    ind_coup, ind_rd = defaultdict(list), defaultdict(list)
    for i in range(len(codes)):
        ind_coup[inds[i]].append(firm_coup[i])
        r = ri(fin.get(codes[i], {}).get("2024"))
        if r is not None:
            ind_rd[inds[i]].append(r)
    ax_b = fig.add_subplot(gs[0, 1])
    xs, ys = [], []
    for ind in ind_coup:
        x, y = np.mean(ind_coup[ind]), np.mean(ind_rd[ind])
        xs.append(x); ys.append(y)
        ax_b.scatter(x, y, s=26, color=C_GNN, edgecolors="white", linewidths=0.4, zorder=3)
        off = {"电子器件": (-0.02, -0.85), "机械行业": (-0.02, 0.4), "仪器仪表": (-0.12, 0.45),
               "生物制药": (0.0, 0.4), "电器行业": (-0.05, -0.9), "汽车制造": (0.0, 0.4),
               "飞机制造": (0.0, -0.9), "船舶制造": (0.02, -0.9),
               "发电设备": (-0.02, 0.4)}.get(ind, (0.0, 0.4))
        ax_b.annotate(IND_EN[ind], (x, y), xytext=(x + off[0], y + off[1]), fontsize=5.6,
                      color="#444444")
    r = np.corrcoef(xs, ys)[0, 1]
    b1, b0 = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs), max(xs), 10)
    ax_b.plot(xr, b0 + b1 * xr, "--", lw=0.8, color=C_MAIN, zorder=2)
    ax_b.text(0.05, 0.9, f"r = {r:.2f}", transform=ax_b.transAxes, fontsize=7, color=C_MAIN)
    ax_b.set_xlabel("Industry mean coupling to innovation policies")
    ax_b.set_ylabel("Industry mean R&D intensity 2024 (%)")

    # --- panel c: GRINM vignette
    ax_c = fig.add_subplot(gs[1, 1])
    yrs = list(range(2019, 2026))
    d = fin.get("600206", {})
    vals = [ri(d.get(str(y))) for y in yrs]
    ax_c.plot(yrs, vals, "-o", ms=4, lw=1.3, color=C_GNN, markeredgecolor="white",
              markeredgewidth=0.4, zorder=3)
    for y, lab, ty in [(2023, "Bio-based\nMaterials Action", 0.72), (2024, "Rare-Earth Reg.;\nKey R&D Program", 0.30)]:
        ax_c.axvline(y, lw=0.7, ls="--", color=C_MAIN, alpha=0.6, zorder=1)
        ax_c.text((y - 2018.7) / (2025.3 - 2018.7) - 0.29, ty, lab, fontsize=5.4,
                  color=C_MAIN, va="bottom", transform=ax_c.transAxes)
    ax_c.set_xlabel("Year")
    ax_c.set_ylabel("R&D intensity (%)")
    ax_c.set_title("GRINM Advanced Materials (top-coupled firm)", fontsize=7)
    ax_c.set_xticks(yrs)

    for ax, letter, xo in [(ax_a, "A", -0.05), (ax_b, "B", -0.18), (ax_c, "C", -0.18)]:
        ax.text(xo, 1.04, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold")
    fig.savefig(f"{FIG}/fig_case_network.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_case_network.png", dpi=400, bbox_inches="tight")
    print("edges drawn:", len(edges), "policies:", len(pols), "firms:", len(firms), "industry r:", round(r, 3))

if __name__ == "__main__":
    main()
