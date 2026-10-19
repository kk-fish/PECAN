# -*- coding: utf-8 -*-
"""Train PECAN and baselines on yearly policy-firm graphs (GPU)."""
import argparse, json, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.decomposition import PCA

DATA = "data"
RES = "results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_YEARS = [2019, 2020, 2021, 2022]
VAL_YEARS = [2023]
TEST_YEARS = [2024]
REL_FILTER = -1.0
TAG = ""

def load(horizon=1):
    suffix = "" if horizon == 1 else f"_h{horizon}"
    g = json.load(open(f"{DATA}/graphs{suffix}.json"))
    f_emb = np.load(f"{DATA}/graph_firm_emb.npy").astype(np.float32)
    p_emb_all = np.load(f"{DATA}/policy_emb.npy").astype(np.float32)
    # standardize firm_x on train years
    trx = np.concatenate([np.array(g["years"][str(t)]["firm_x"], dtype=np.float32) for t in TRAIN_YEARS])
    mu, sd = trx.mean(0), trx.std(0) + 1e-6
    # residual target stats: delta = y_{t+1} - rd_int_t, computed on train years
    deltas = []
    for t in TRAIN_YEARS:
        d = g["years"][str(t)]
        m = np.array(d["mask"])
        deltas.append((np.array(d["y"]) - np.array(d["firm_x"], dtype=np.float32)[:, 3])[m])
    deltas = np.concatenate(deltas)
    y_mu, y_sd = float(deltas.mean()), float(deltas.std() + 1e-6)
    years = {}
    for t, d in g["years"].items():
        fxr = np.array(d["firm_x"], dtype=np.float32)
        rd_now = fxr[:, 3].copy()
        fx = (fxr - mu) / sd
        pf = np.array(d["policy_feat"], dtype=np.float32) if d["policy_feat"] else np.zeros((0, 30), np.float32)
        pe = p_emb_all[d["policy_emb_idx"]] if d["policy_emb_idx"] else np.zeros((0, p_emb_all.shape[1]), np.float32)
        years[int(t)] = {
            "fx": torch.tensor(fx), "pf": torch.tensor(pf), "pe": torch.tensor(pe),
            "pf_edge": torch.tensor([d["pf_src"], d["pf_dst"]], dtype=torch.long),
            "pf_attr": torch.tensor(np.array(d["pf_attr"], dtype=np.float32) if d["pf_attr"] else np.zeros((0, 4), np.float32)),
            "y": torch.tensor(np.array(d["y"], dtype=np.float32)),
            "rd_now": torch.tensor(rd_now),
            "yc": torch.tensor(np.array(d["y_cls"], dtype=np.float32)),
            "mask": torch.tensor(np.array(d["mask"], dtype=bool)),
            "policy_urls": d["policy_urls"],
        }
    ff = torch.tensor([g["ff_src"], g["ff_dst"]], dtype=torch.long)
    meta = {"codes": g["firm_codes"], "names": g["firm_names"], "industry": g["firm_industry"],
            "y_mu": y_mu, "y_sd": y_sd}
    return years, torch.tensor(f_emb), ff, meta

class PECAN(nn.Module):
    def __init__(self, femb_dim, fx_dim, pfeat_dim, pemb_dim, hid=128, heads=4,
                 use_policy=True, use_llm=True, use_ff=True):
        super().__init__()
        self.use_policy, self.use_llm, self.use_ff = use_policy, use_llm, use_ff
        self.f_enc = nn.Sequential(nn.Linear(femb_dim + fx_dim, hid), nn.ReLU(), nn.LayerNorm(hid))
        p_in = pemb_dim + (pfeat_dim if use_llm else 0)
        self.p_enc = nn.Sequential(nn.Linear(p_in, hid), nn.ReLU(), nn.LayerNorm(hid))
        self.pf_gat = GATConv((hid, hid), hid // heads, heads=heads, add_self_loops=False, edge_dim=4)
        self.ff_gat1 = GATConv(hid, hid // heads, heads=heads)
        self.ff_gat2 = GATConv(hid, hid // heads, heads=heads)
        self.gate = nn.Sequential(nn.Linear(hid * 2, hid), nn.Sigmoid())
        self.head_reg = nn.Sequential(nn.Linear(hid * 2 + fx_dim, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1))
        self.head_cls = nn.Sequential(nn.Linear(hid + fx_dim, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1))

    def forward(self, femb, fx, pfeat, pemb, pf_edge, pf_attr, ff_edge, return_att=False):
        h_f = self.f_enc(torch.cat([femb, fx], -1))
        att = None
        if self.use_policy and pemb.shape[0] > 0 and pf_edge.shape[1] > 0:
            p_in = torch.cat([pemb, pfeat], -1) if self.use_llm else pemb
            h_p = self.p_enc(p_in)
            if return_att:
                h_pf, (eidx, alpha) = self.pf_gat((h_p, h_f), pf_edge, edge_attr=pf_attr,
                                                  return_attention_weights=True)
                att = (eidx, alpha.mean(-1))
            else:
                h_pf = self.pf_gat((h_p, h_f), pf_edge, edge_attr=pf_attr)
            g = self.gate(torch.cat([h_f, h_pf], -1))
            h1 = h_f + g * F.relu(h_pf)
        else:
            h1 = h_f
        if self.use_ff:
            h2 = h1 + F.relu(self.ff_gat1(h1, ff_edge))
            h3 = h2 + F.relu(self.ff_gat2(h2, ff_edge))
        else:
            h3 = h1
        z = torch.cat([h_f, h3, fx], -1)
        z_cls = torch.cat([h_f, fx], -1)
        if return_att:
            self._last_h = torch.cat([h_f, h3], -1).detach()
        return self.head_reg(z).squeeze(-1), self.head_cls(z_cls).squeeze(-1), att

class FirmGNN(nn.Module):
    """Homogeneous firm-graph baselines: gcn | gat | sage | mlp."""
    def __init__(self, femb_dim, fx_dim, kind, hid=128):
        super().__init__()
        self.kind = kind
        self.enc = nn.Sequential(nn.Linear(femb_dim + fx_dim, hid), nn.ReLU(), nn.LayerNorm(hid))
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(hid, hid), GCNConv(hid, hid)
        elif kind == "gat":
            self.c1, self.c2 = GATConv(hid, hid // 4, heads=4), GATConv(hid, hid // 4, heads=4)
        elif kind == "sage":
            self.c1, self.c2 = SAGEConv(hid, hid), SAGEConv(hid, hid)
        self.head_reg = nn.Sequential(nn.Linear(hid + fx_dim, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1))
        self.head_cls = nn.Sequential(nn.Linear(hid + fx_dim, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1))

    def forward(self, femb, fx, ff_edge):
        h = self.enc(torch.cat([femb, fx], -1))
        if self.kind != "mlp":
            h = h + F.relu(self.c1(h, ff_edge))
            h = h + F.relu(self.c2(h, ff_edge))
        z = torch.cat([h, fx], -1)
        return self.head_reg(z).squeeze(-1), self.head_cls(z).squeeze(-1)

def metrics_reg(y, p):
    y, p = np.asarray(y), np.asarray(p)
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    mae = float(np.abs(y - p).mean())
    ss = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else 0.0
    return rmse, mae, r2

def metrics_cls(y, p):
    y, p = np.asarray(y), np.asarray(p)
    try:
        auc = float(roc_auc_score(y, p))
    except Exception:
        auc = 0.5
    lab = (p > 0.5).astype(int)
    return auc, float(f1_score(y, lab)), float(accuracy_score(y, lab))

def run_torch_model(name, years, femb, ff, meta, seed, epochs=400, lr=1e-3, return_artifacts=False):
    torch.manual_seed(seed); np.random.seed(seed)
    femb_d, fx_d = femb.shape[1], years[2019]["fx"].shape[1]
    pf_d = 30
    pemb_d = years[2021]["pe"].shape[1]
    if name.startswith("pecan"):
        model = PECAN(femb_d, fx_d, pf_d, pemb_d,
                      use_policy=(name != "pecan_no_policy"),
                      use_llm=(name != "pecan_no_llm"),
                      use_ff=(name != "pecan_no_ff")).to(DEV)
    else:
        model = FirmGNN(femb_d, fx_d, name).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    femb = femb.to(DEV); ff_e = ff.to(DEV)
    y_mu, y_sd = meta["y_mu"], meta["y_sd"]
    for t in years:
        for k in ["fx", "pf", "pe", "pf_edge", "pf_attr", "y", "rd_now", "yc", "mask"]:
            years[t][k] = years[t][k].to(DEV)

    def fwd(t, return_att=False):
        d = years[t]
        edge = d["pf_edge"]
        attr = d["pf_attr"]
        if name == "pecan_no_semantic" and attr.shape[0] > 0:
            keep = attr[:, 0] > 0.5
            edge, attr = edge[:, keep], attr[keep]
        if REL_FILTER > 0 and name.startswith("pecan") and name != "pecan_no_llm" and attr.shape[0] > 0:
            keep = attr[:, 2] >= REL_FILTER
            edge, attr = edge[:, keep], attr[keep]
        if isinstance(model, PECAN):
            return model(femb, d["fx"], d["pf"], d["pe"], edge, attr, ff_e, return_att)
        r, c = model(femb, d["fx"], ff_e)
        return r, c, None

    def evaluate(ts):
        model.eval()
        ys, ps, ycs, pcs = [], [], [], []
        with torch.no_grad():
            for t in ts:
                d = years[t]
                r, c, _ = fwd(t)
                m = d["mask"]
                ys += (d["y"][m]).tolist()
                ps += (d["rd_now"][m] + r[m] * y_sd + y_mu).clamp(0, 50).tolist()
                ycs += d["yc"][m].tolist()
                pcs += torch.sigmoid(c[m]).tolist()
        return metrics_reg(ys, ps), metrics_cls(ycs, pcs)

    best_val, best_state, patience = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = 0
        for t in TRAIN_YEARS:
            d = years[t]
            r, c, _ = fwd(t)
            m = d["mask"]
            yn = (d["y"][m] - d["rd_now"][m] - y_mu) / y_sd
            loss = loss + F.huber_loss(r[m], yn) + 1.0 * F.binary_cross_entropy_with_logits(c[m], d["yc"][m])
        loss.backward()
        opt.step()
        if ep % 5 == 0:
            (vr, _, _), _ = evaluate(VAL_YEARS)
            if vr < best_val - 1e-4:
                best_val, patience = vr, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience > 20:
                    break
    if best_state:
        model.load_state_dict(best_state)
    (rmse, mae, r2), (auc, f1, acc) = evaluate(TEST_YEARS)
    (v_rmse, _, _), (v_auc, _, _) = evaluate(VAL_YEARS)

    if return_artifacts:
        return dict(model=name, seed=seed, rmse=rmse, mae=mae, r2=r2, auc=auc, f1=f1, acc=acc,
                    val_rmse=v_rmse, val_auc=v_auc), model, fwd

    result = dict(model=name, seed=seed, rmse=rmse, mae=mae, r2=r2, auc=auc, f1=f1, acc=acc,
                  val_rmse=v_rmse, val_auc=v_auc)
    # dump attention + predictions for case study (full pecan, seed 0)
    if name == "pecan" and seed == 0:
        model.eval()
        with torch.no_grad():
            t = TEST_YEARS[0]
            r, c, att = fwd(t, return_att=True)
            d = years[t]
            m = d["mask"]
            preds = (d["rd_now"] + r * y_sd + y_mu).clamp(0, 50).cpu().numpy()
            json.dump({"y": d["y"][m].cpu().tolist(),
                       "pred": preds[m.cpu().numpy()].tolist(),
                       "industry": [meta["industry"][i] for i in range(len(m)) if m[i]]},
                      open(f"{RES}/preds_test{TAG}.json", "w"))
            if att is not None:
                eidx, alpha = att
                json.dump({"src": eidx[0].cpu().tolist(), "dst": eidx[1].cpu().tolist(),
                           "alpha": alpha.cpu().tolist(),
                           "policy_urls": d["policy_urls"],
                           "codes": meta["codes"], "names": meta["names"],
                           "industry": meta["industry"]},
                          open(f"{RES}/attention_test{TAG}.json", "w"))
    return result

def run_xgb(years, femb, meta, seed, use_expo=True, extra_feats=None, name="xgboost"):
    import xgboost as xgb
    pca = PCA(n_components=32, random_state=0)
    emb32 = pca.fit_transform(femb.cpu().numpy())
    def feats(t):
        d = years[t]
        n = d["fx"].shape[0]
        cols = [d["fx"].cpu().numpy(), emb32]
        if use_expo:
            expo = np.zeros((n, 3), np.float32)
            edge = d["pf_edge"].cpu().numpy()
            attr = d["pf_attr"].cpu().numpy()
            if edge.shape[1] > 0:
                for j in range(edge.shape[1]):
                    fi = edge[1, j]
                    expo[fi, 0] += 1
                    expo[fi, 1] += attr[j, 1]
                    expo[fi, 2] += attr[j, 0]
                expo[:, 1] /= np.maximum(expo[:, 0], 1)
            cols.append(expo)
        if extra_feats is not None:
            cols.append(extra_feats[t])
        return np.concatenate(cols, 1)
    Xtr = np.concatenate([feats(t)[years[t]["mask"].cpu().numpy()] for t in TRAIN_YEARS + VAL_YEARS])
    ytr = np.concatenate([years[t]["y"][years[t]["mask"]].cpu().numpy() for t in TRAIN_YEARS + VAL_YEARS])
    yctr = np.concatenate([years[t]["yc"][years[t]["mask"]].cpu().numpy() for t in TRAIN_YEARS + VAL_YEARS])
    t = TEST_YEARS[0]
    Xte = feats(t)[years[t]["mask"].cpu().numpy()]
    yte = years[t]["y"][years[t]["mask"]].cpu().numpy()
    ycte = years[t]["yc"][years[t]["mask"]].cpu().numpy()
    reg = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.8, random_state=seed)
    reg.fit(Xtr, ytr)
    cls = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.8, random_state=seed,
                            eval_metric="logloss")
    cls.fit(Xtr, yctr)
    rmse, mae, r2 = metrics_reg(yte, reg.predict(Xte))
    auc, f1, acc = metrics_cls(ycte, cls.predict_proba(Xte)[:, 1])
    if name == "pecan_hybrid" and seed == 0:
        t = TEST_YEARS[0]
        m = years[t]["mask"].cpu().numpy()
        json.dump({"y": yte.tolist(), "pred": reg.predict(Xte).tolist(),
                   "industry": [meta["industry"][i] for i in range(len(m)) if m[i]]},
                  open(f"{RES}/preds_hybrid{TAG}.json", "w"))
    return dict(model=name, seed=seed, rmse=rmse, mae=mae, r2=r2, auc=auc, f1=f1, acc=acc)

def run_ensemble(dual_name, years, femb, ff, meta, seed):
    """PECAN dual-channel: graph-attention neural channel + GBDT channel, output fusion tuned on val."""
    import xgboost as xgb
    neural_name = dual_name.replace("pecan_dual", "pecan")
    _, model, fwd = run_torch_model(neural_name, years, femb, ff, meta, seed, return_artifacts=True)
    model.eval()
    y_mu, y_sd = meta["y_mu"], meta["y_sd"]

    def neural_preds(t):
        with torch.no_grad():
            r, c, att = fwd(t, return_att=True)
            d = years[t]
            pr = (d["rd_now"] + r * y_sd + y_mu).clamp(0, 50).cpu().numpy()
            pc = torch.sigmoid(c).cpu().numpy()
        return pr, pc, att

    # tree channel
    use_expo = "no_policy" not in dual_name
    pca = PCA(n_components=32, random_state=0)
    emb32 = pca.fit_transform(femb.cpu().numpy())

    def tree_feats(t):
        d = years[t]
        n = d["fx"].shape[0]
        cols = [d["fx"].cpu().numpy(), emb32]
        if use_expo:
            expo = np.zeros((n, 3), np.float32)
            edge = d["pf_edge"].cpu().numpy()
            attr = d["pf_attr"].cpu().numpy()
            for j in range(edge.shape[1]):
                fi = edge[1, j]
                expo[fi, 0] += 1
                expo[fi, 1] += attr[j, 1]
                expo[fi, 2] += attr[j, 0]
            expo[:, 1] /= np.maximum(expo[:, 0], 1)
            cols.append(expo)
        return np.concatenate(cols, 1)

    def stack(ts):
        X = np.concatenate([tree_feats(t)[years[t]["mask"].cpu().numpy()] for t in ts])
        y = np.concatenate([years[t]["y"][years[t]["mask"]].cpu().numpy() for t in ts])
        yc = np.concatenate([years[t]["yc"][years[t]["mask"]].cpu().numpy() for t in ts])
        return X, y, yc

    Xtr, ytr, yctr = stack(TRAIN_YEARS)
    reg = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.8, random_state=seed)
    reg.fit(Xtr, ytr)
    cls = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.8, random_state=seed,
                            eval_metric="logloss")
    cls.fit(Xtr, yctr)

    def fused(t, wr, wc):
        d = years[t]
        m = d["mask"].cpu().numpy()
        pr_n, pc_n, att = neural_preds(t)
        X = tree_feats(t)[m]
        pr = wr * pr_n[m] + (1 - wr) * reg.predict(X)
        pc = wc * pc_n[m] + (1 - wc) * cls.predict_proba(X)[:, 1]
        return d["y"][d["mask"]].cpu().numpy(), pr, d["yc"][d["mask"]].cpu().numpy(), pc, att, m

    # tune fusion weights on validation year
    grid = np.linspace(0, 1, 11)
    best_wr, best_rmse, best_wc, best_auc = 0.5, 1e9, 0.5, 0
    yv, _, ycv, _, _, _ = fused(VAL_YEARS[0], 0.5, 0.5)
    pr_n_v, pc_n_v, _ = neural_preds(VAL_YEARS[0])
    mv = years[VAL_YEARS[0]]["mask"].cpu().numpy()
    Xv = tree_feats(VAL_YEARS[0])[mv]
    pr_t_v, pc_t_v = reg.predict(Xv), cls.predict_proba(Xv)[:, 1]
    for w in grid:
        rmse_v = np.sqrt((((w * pr_n_v[mv] + (1 - w) * pr_t_v) - yv) ** 2).mean())
        if rmse_v < best_rmse:
            best_rmse, best_wr = rmse_v, w
        try:
            auc_v = roc_auc_score(ycv, w * pc_n_v[mv] + (1 - w) * pc_t_v)
        except Exception:
            auc_v = 0.5
        if auc_v > best_auc:
            best_auc, best_wc = auc_v, w

    t = TEST_YEARS[0]
    yte, pr, ycte, pc, att, m = fused(t, best_wr, best_wc)
    rmse, mae, r2 = metrics_reg(yte, pr)
    auc, f1, acc = metrics_cls(ycte, pc)
    if dual_name == "pecan_dual" and seed == 0:
        json.dump({"y": yte.tolist(), "pred": pr.tolist(), "wr": best_wr, "wc": best_wc,
                   "industry": [meta["industry"][i] for i in range(len(m)) if m[i]]},
                  open(f"{RES}/preds_dual{TAG}.json", "w"))
        if att is not None:
            eidx, alpha = att
            d = years[t]
            json.dump({"src": eidx[0].cpu().tolist(), "dst": eidx[1].cpu().tolist(),
                       "alpha": alpha.cpu().tolist(), "policy_urls": d["policy_urls"],
                       "codes": meta["codes"], "names": meta["names"],
                       "industry": meta["industry"]},
                      open(f"{RES}/attention_dual{TAG}.json", "w"))
    return dict(model=dual_name, seed=seed, rmse=rmse, mae=mae, r2=r2, auc=auc, f1=f1, acc=acc,
                wr=best_wr, wc=best_wc, val_rmse=best_rmse, val_auc=best_auc)

def run_hybrid(years, femb, ff, meta, seed):
    """PECAN-hybrid: GNN coupling representation + gradient boosting head."""
    _, model, fwd = run_torch_model("pecan", years, femb, ff, meta, seed, return_artifacts=True)
    model.eval()
    reps, coups = {}, {}
    with torch.no_grad():
        for t in years:
            _, _, att = fwd(t, return_att=True)
            h = model._last_h.cpu().numpy()
            d = years[t]
            n = h.shape[0]
            coup = np.zeros((n, 33), np.float32)
            if att is not None:
                eidx, alpha = att
                src = eidx[0].cpu().numpy(); dst = eidx[1].cpu().numpy()
                al = alpha.cpu().numpy()
                pf = d["pf"].cpu().numpy()
                sums = np.zeros(n); maxs = np.zeros(n); cnts = np.zeros(n)
                wpol = np.zeros((n, 30))
                for j in range(len(dst)):
                    i = dst[j]; a = float(al[j])
                    sums[i] += a
                    if a > maxs[i]:
                        maxs[i] = a
                    cnts[i] += 1
                    wpol[i] += a * pf[src[j]]
                wpol = wpol / np.maximum(sums, 1e-8)[:, None]
                coup[:, 0] = sums; coup[:, 1] = maxs; coup[:, 2] = cnts
                coup[:, 3:] = wpol
            reps[t] = h; coups[t] = coup
    tr_h = np.concatenate([reps[t] for t in TRAIN_YEARS])
    pca_h = PCA(n_components=32, random_state=0).fit(tr_h)
    extra = {t: np.concatenate([pca_h.transform(reps[t]), coups[t]], 1) for t in years}
    return run_xgb(years, femb, meta, seed, use_expo=True, extra_feats=extra, name="pecan_hybrid")

def main():
    global TRAIN_YEARS, VAL_YEARS, TEST_YEARS, REL_FILTER, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pecan,pecan_no_llm,pecan_no_policy,pecan_no_semantic,pecan_no_ff,mlp,gcn,gat,sage,xgboost")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--relfilter", type=float, default=-1.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    H = args.horizon
    TRAIN_YEARS = list(range(2019, 2024 - H))
    VAL_YEARS = [2024 - H]
    TEST_YEARS = [2025 - H]
    REL_FILTER = args.relfilter
    TAG = args.tag
    print("years", TRAIN_YEARS, VAL_YEARS, TEST_YEARS, "relfilter", REL_FILTER, flush=True)
    os.makedirs(RES, exist_ok=True)
    years, femb, ff, meta = load(H)
    rows = []
    for m in args.models.split(","):
        for s in range(args.seeds):
            if m == "xgboost":
                r = run_xgb(years, femb, meta, s)
            elif m == "xgb_nopolicy":
                r = run_xgb(years, femb, meta, s, use_expo=False, name="xgb_nopolicy")
            elif m == "pecan_hybrid":
                r = run_hybrid(years, femb, ff, meta, s)
            elif m.startswith("pecan_dual"):
                r = run_ensemble(m, years, femb, ff, meta, s)
            else:
                r = run_torch_model(m, years, femb, ff, meta, s)
            print(r, flush=True)
            rows.append(r)
            import pandas as pd
            pd.DataFrame(rows).to_csv(f"{RES}/results{TAG}.csv", index=False)
    print("TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()
