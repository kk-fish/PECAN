# -*- coding: utf-8 -*-
"""Crawl industrial policy documents from gov.cn policy library."""
import requests, json, time, os, re, random

OUT = "data/policies.jsonl"
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

KEYWORDS = [
    "智能制造", "人工智能", "集成电路", "新能源汽车", "生物医药",
    "高端装备", "工业互联网", "新材料", "数字经济", "机器人",
    "软件产业", "大数据", "新能源", "节能环保", "信息技术",
    "先进制造", "医疗器械", "半导体", "数字化转型", "专精特新",
]

def search(t, q, page, searchfield=""):
    params = {"t": t, "q": q, "timetype": "timeqb", "sort": "score", "sortType": "1",
              "searchfield": searchfield, "p": str(page), "n": "20",
              "inpro": "", "dup": "", "orpro": "", "type": "gwyzcwjk"}
    r = requests.get("https://sousuo.www.gov.cn/search-gov/data", params=params, timeout=20, headers=H)
    d = r.json()
    if d.get("code") != 200 or not d.get("searchVO"):
        return []
    return d["searchVO"].get("listVO") or []

def fetch_text(url):
    try:
        r = requests.get(url, timeout=25, headers=H)
        r.encoding = r.apparent_encoding or "utf-8"
        html = r.text
    except Exception:
        return ""
    m = re.search(r'<div[^>]*id="UCAP-CONTENT"[^>]*>(.*?)</div>\s*<!--', html, re.S)
    if not m:
        m = re.search(r'<div[^>]*id="UCAP-CONTENT"[^>]*>(.*)', html, re.S)
    body = m.group(1) if m else html
    paras = re.findall(r"<p[^>]*>(.*?)</p>", body, re.S)
    txts = []
    for p in paras:
        p = re.sub(r"<[^>]+>", "", p)
        p = re.sub(r"&nbsp;?", " ", p).replace("&ldquo;", "“").replace("&rdquo;", "”")
        p = re.sub(r"&[a-z]+;", "", p).strip()
        if p:
            txts.append(p)
    return "\n".join(txts)

def main():
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["url"])
                except Exception:
                    pass
    fout = open(OUT, "a", encoding="utf-8")
    n_new = 0
    for kw in KEYWORDS:
        for t, sf, maxp in [("zhengcelibrary_gw", "", 3), ("zhengcelibrary_bm", "title", 3)]:
            for page in range(1, maxp + 1):
                try:
                    items = search(t, kw, page, sf)
                except Exception as e:
                    print("search err", kw, t, page, str(e)[:100], flush=True)
                    time.sleep(5)
                    continue
                if not items:
                    break
                for it in items:
                    url = it.get("url", "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    pub = it.get("pubtimeStr", "") or str(it.get("pubtime", ""))
                    title = re.sub(r"</?em>", "", it.get("title", ""))
                    text = fetch_text(url)
                    time.sleep(random.uniform(0.6, 1.2))
                    if len(text) < 400:
                        continue
                    rec = {"title": title, "url": url, "pubtime": pub,
                           "source": it.get("source", ""), "fwzh": it.get("fwzh", ""),
                           "keyword": kw, "lib": t, "text": text[:20000]}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_new += 1
                time.sleep(random.uniform(0.8, 1.5))
        print(f"[{kw}] cumulative new: {n_new}", flush=True)
    fout.close()
    print("DONE, new:", n_new, flush=True)

if __name__ == "__main__":
    main()
