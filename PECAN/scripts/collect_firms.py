# -*- coding: utf-8 -*-
"""Collect A-share firm data: industry boards, business text, yearly financials."""
import akshare as ak
import pandas as pd
import json, time, os, random, traceback

DATA = "data"
SECTORS = {
    "new_dzxx": "电子信息", "new_dzqj": "电子器件", "new_jxhy": "机械行业",
    "new_qczz": "汽车制造", "new_swzz": "生物制药", "new_ylqx": "医疗器械",
    "new_dqhy": "电器行业", "new_fdsb": "发电设备", "new_fjzz": "飞机制造",
    "new_yqyb": "仪器仪表", "new_hbhy": "环保行业", "new_hghy": "化工行业",
    "new_cbzz": "船舶制造",
}
PER_SECTOR = 130  # top N by market cap

def get_universe():
    rows = []
    for label, name in SECTORS.items():
        for attempt in range(3):
            try:
                df = ak.stock_sector_detail(sector=label)
                df = df[~df["code"].astype(str).str.startswith(("8", "4", "9"))]
                df = df.sort_values("mktcap", ascending=False).head(PER_SECTOR)
                for _, r in df.iterrows():
                    rows.append({"code": str(r["code"]).zfill(6), "name": r["name"], "industry": name})
                print(name, len(df), flush=True)
                break
            except Exception as e:
                print("sector err", name, str(e)[:80], flush=True)
                time.sleep(5)
        time.sleep(random.uniform(1, 2))
    uni = pd.DataFrame(rows).drop_duplicates(subset="code")
    uni.to_csv(f"{DATA}/universe.csv", index=False)
    print("universe:", len(uni), flush=True)
    return uni

def prefix(code):
    return ("SH" if code.startswith(("6", "9")) else "SZ") + code

def main():
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(f"{DATA}/universe.csv"):
        uni = pd.read_csv(f"{DATA}/universe.csv", dtype={"code": str})
    else:
        uni = get_universe()

    # resume support
    done_txt, done_fin = set(), set()
    if os.path.exists(f"{DATA}/firm_text.jsonl"):
        for line in open(f"{DATA}/firm_text.jsonl"):
            try:
                done_txt.add(json.loads(line)["code"])
            except Exception:
                pass
    if os.path.exists(f"{DATA}/firm_fin.jsonl"):
        for line in open(f"{DATA}/firm_fin.jsonl"):
            try:
                done_fin.add(json.loads(line)["code"])
            except Exception:
                pass

    ftxt = open(f"{DATA}/firm_text.jsonl", "a", encoding="utf-8")
    ffin = open(f"{DATA}/firm_fin.jsonl", "a", encoding="utf-8")
    for i, (_, row) in enumerate(uni.iterrows()):
        code = str(row["code"]).zfill(6)
        # business text
        if code not in done_txt:
            try:
                z = ak.stock_zyjs_ths(symbol=code)
                d = z.iloc[0].to_dict()
                rec = {"code": code, "name": row["name"], "industry": row["industry"],
                       "main_business": str(d.get("主营业务", "")),
                       "product_type": str(d.get("产品类型", "")),
                       "scope": str(d.get("经营范围", ""))}
                ftxt.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ftxt.flush()
            except Exception as e:
                print("zyjs err", code, str(e)[:60], flush=True)
            time.sleep(random.uniform(0.4, 0.9))
        # yearly financials
        if code not in done_fin:
            try:
                df = ak.stock_profit_sheet_by_yearly_em(symbol=prefix(code))
                keep = {}
                for _, r in df.iterrows():
                    year = str(r["REPORT_DATE"])[:4]
                    if year < "2018":
                        continue
                    keep[year] = {
                        "revenue": float(r.get("TOTAL_OPERATE_INCOME") or 0),
                        "rd": float(r.get("RESEARCH_EXPENSE") or 0),
                        "netprofit": float(r.get("PARENT_NETPROFIT") or r.get("NETPROFIT") or 0),
                        "operate_cost": float(r.get("TOTAL_OPERATE_COST") or 0),
                    }
                ffin.write(json.dumps({"code": code, "fin": keep}, ensure_ascii=False) + "\n")
                ffin.flush()
            except Exception as e:
                print("fin err", code, str(e)[:60], flush=True)
            time.sleep(random.uniform(0.4, 0.9))
        if i % 25 == 0:
            print(f"progress {i}/{len(uni)}", flush=True)
    ftxt.close(); ffin.close()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
