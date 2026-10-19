# -*- coding: utf-8 -*-
"""Build yearly policy-firm heterogeneous graphs (CPU, jump server)."""
import json, math, os
import numpy as np
import pandas as pd

DATA = "data"
INDUSTRIES = ["电子信息", "电子器件", "机械行业", "汽车制造", "生物制药",
              "医疗器械", "电器行业", "发电设备", "飞机制造", "仪器仪表",
              "环保行业", "化工行业", "船舶制造"]
TOOLS = ["财政补贴", "税收优惠", "金融支持", "政府采购", "人才支持",
         "技术研发支持", "基础设施建设", "标准与规范", "市场准入", "试点示范", "监管约束"]
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--horizon", type=int, default=1)
ARGS = _ap.parse_args()
H = ARGS.horizon
YEARS = list(range(2019, 2026 - H))  # feature year t -> target t+H
POLICY_WINDOW = 3  # policies published in [t-2, t]
SEM_TOPK_PF = 15   # semantic policy->firm edges per policy
SEM_TOPK_FF = 5    # semantic firm->firm edges

def main():
    firms = [json.loads(l) for l in open(f"{DATA}/firm_text.jsonl")]
    firm_order = json.load(open(f"{DATA}/firm_order.json"))
    f_emb = np.load(f"{DATA}/firm_emb.npy")
    code2idx = {c: i for i, c in enumerate(firm_order)}
    firms = [f for f in firms if f["code"] in code2idx]
    firms.sort(key=lambda f: code2idx[f["code"]])

    fin = {}
    for l in open(f"{DATA}/firm_fin.jsonl"):
        d = json.loads(l)
        fin[d["code"]] = d["fin"]

    policies = [json.loads(l) for l in open(f"{DATA}/policies.jsonl")]
    p_order = json.load(open(f"{DATA}/policy_order.json"))
    p_emb = np.load(f"{DATA}/policy_emb.npy")
    url2idx = {u: i for i, u in enumerate(p_order)}
    llm = {}
    for l in open(f"{DATA}/policy_llm.jsonl"):
        d = json.loads(l)
        llm[d["url"]] = d

    # policy features
    def policy_feat(p):
        a = llm.get(p["url"], {})
        tools = a.get("policy_tools") or []
        tinds = a.get("target_industries") or []
        v = [1.0 if t in tools else 0.0 for t in TOOLS]
        v += [1.0 if i in tinds else 0.0 for i in INDUSTRIES]
        def num(x, default):
            try:
                return float(x)
            except Exception:
                return default
        v += [num(a.get("binding_strength"), 3) / 5.0,
              num(a.get("innovation_relevance"), 0.5),
              num(a.get("supply_side"), 0.33), num(a.get("demand_side"), 0.33),
              num(a.get("environment_side"), 0.33),
              1.0 if p["lib"] == "zhengcelibrary_gw" else 0.0]
        return v, tinds

    p_year = {}
    for p in policies:
        y = str(p.get("pubtime", ""))[:4]
        p_year[p["url"]] = int(y) if y.isdigit() else None

    ind2idx = {ind: i for i, ind in enumerate(INDUSTRIES)}
    firm_ind = np.array([ind2idx.get(f["industry"], -1) for f in firms])

    # static firm-firm edges: within-industry semantic kNN (k=10)
    ff_src, ff_dst = [], []
    for ind in range(len(INDUSTRIES)):
        idxs = np.where(firm_ind == ind)[0]
        if len(idxs) < 2:
            continue
        sub = f_emb[idxs] @ f_emb[idxs].T
        np.fill_diagonal(sub, -1)
        k = min(10, len(idxs) - 1)
        for ai, a in enumerate(idxs):
            for bi in np.argsort(-sub[ai])[:k]:
                ff_src.append(int(a)); ff_dst.append(int(idxs[bi]))
    # semantic firm-firm kNN (cross-industry allowed)
    sim_ff = f_emb @ f_emb.T
    np.fill_diagonal(sim_ff, -1)
    for a in range(len(firms)):
        for b in np.argsort(-sim_ff[a])[:SEM_TOPK_FF]:
            ff_src.append(a); ff_dst.append(int(b))
    ff = pd.DataFrame({"src": ff_src, "dst": ff_dst}).drop_duplicates()

    # firm yearly features and targets
    def val(d, k):
        if not d:
            return 0.0
        v = d.get(k, 0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if (math.isnan(v) or math.isinf(v)) else v

    def fin_feats(code, t):
        d = fin.get(code, {})
        cur, prev, nxt = d.get(str(t)), d.get(str(t - 1)), d.get(str(t + H))
        if not cur or val(cur, "revenue") <= 0:
            return None, None, None
        rev = val(cur, "revenue")
        rd_int = 100.0 * val(cur, "rd") / rev
        growth = (rev / val(prev, "revenue") - 1.0) if val(prev, "revenue") > 0 else 0.0
        margin = val(cur, "netprofit") / rev
        rd_prev_int = 100.0 * val(prev, "rd") / val(prev, "revenue") if val(prev, "revenue") > 0 else rd_int
        x = [np.log10(rev + 1), np.clip(growth, -1, 3), np.clip(margin, -1, 1),
             np.clip(rd_int, 0, 50), np.clip(rd_int - rd_prev_int, -20, 20)]
        y = None
        y_cls = None
        if nxt and val(nxt, "revenue") > 0:
            y = float(np.clip(100.0 * val(nxt, "rd") / val(nxt, "revenue"), 0, 50))
            y_cls = 1 if val(nxt, "rd") > val(cur, "rd") else 0
        return x, y, y_cls

    out = {"industries": INDUSTRIES, "tools": TOOLS, "years": {}}
    np.save(f"{DATA}/graph_firm_emb.npy", f_emb)

    for t in YEARS:
        # active policies
        act = [p for p in policies if p_year.get(p["url"]) and t - POLICY_WINDOW < p_year[p["url"]] <= t]
        pf_feats, p_urls, p_tinds = [], [], []
        for p in act:
            v, tinds = policy_feat(p)
            pf_feats.append(v); p_urls.append(p["url"]); p_tinds.append(tinds)
        p_idx = [url2idx[u] for u in p_urls]
        pe = p_emb[p_idx] if p_idx else np.zeros((0, p_emb.shape[1]))

        # policy->firm edges; edge attr: [industry_match, cos_sim, innovation_relevance, binding_strength]
        pf_src, pf_dst, pf_attr = [], [], []
        if act:
            sim_pf = pe @ f_emb.T  # [P, F]
            for pi, p in enumerate(act):
                rel = pf_feats[pi][-5]   # innovation_relevance
                strength = pf_feats[pi][-6]  # binding_strength/5
                matched = set()
                for fi in range(len(firms)):
                    if firms[fi]["industry"] in p_tinds[pi]:
                        matched.add(fi)
                topk = set(int(x) for x in np.argsort(-sim_pf[pi])[:SEM_TOPK_PF])
                for fi in matched | topk:
                    pf_src.append(pi); pf_dst.append(fi)
                    pf_attr.append([1.0 if fi in matched else 0.0, float(sim_pf[pi, fi]),
                                    float(rel), float(strength)])

        xs, ys, ycs, mask = [], [], [], []
        for f in firms:
            x, y, yc = fin_feats(f["code"], t)
            if x is None:
                xs.append([0] * 5); ys.append(0.0); ycs.append(0); mask.append(False)
            else:
                xs.append(x)
                ys.append(y if y is not None else 0.0)
                ycs.append(yc if yc is not None else 0)
                mask.append(y is not None)

        out["years"][t] = {
            "policy_urls": p_urls, "policy_feat": pf_feats, "policy_emb_idx": p_idx,
            "pf_src": pf_src, "pf_dst": pf_dst, "pf_attr": pf_attr,
            "firm_x": xs, "y": ys, "y_cls": ycs, "mask": mask,
        }
        print(f"year {t}: policies={len(act)} pf_edges={len(pf_src)} labeled_firms={sum(mask)}", flush=True)

    out["ff_src"] = ff["src"].tolist()
    out["ff_dst"] = ff["dst"].tolist()
    out["firm_codes"] = [f["code"] for f in firms]
    out["firm_names"] = [f["name"] for f in firms]
    out["firm_industry"] = [f["industry"] for f in firms]
    suffix = "" if H == 1 else f"_h{H}"
    json.dump(out, open(f"{DATA}/graphs{suffix}.json", "w"), ensure_ascii=False)
    print("BUILD_DONE, ff_edges:", len(ff), flush=True)

if __name__ == "__main__":
    main()
