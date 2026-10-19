# -*- coding: utf-8 -*-
"""Merge crawled policy metadata with the LLM profiles into data/policy_meta.json.

policy_meta.json is a lookup table (url -> title, date, collection, LLM tools/targets/
relevance/binding strength) used by the coupling analysis and the case-study figures.
"""
import json

DATA = "data"


def main():
    llm = {}
    for line in open(f"{DATA}/policy_llm.jsonl", encoding="utf-8"):
        r = json.loads(line)
        llm[r["url"]] = r
    meta = {}
    for line in open(f"{DATA}/policies.jsonl", encoding="utf-8"):
        p = json.loads(line)
        r = llm.get(p["url"], {})
        meta[p["url"]] = {"title": p["title"], "pubtime": p["pubtime"], "lib": p["lib"],
                          "tools": r.get("policy_tools", []), "industries": r.get("target_industries", []),
                          "relevance": r.get("innovation_relevance"), "strength": r.get("binding_strength")}
    json.dump(meta, open(f"{DATA}/policy_meta.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("policies:", len(meta))


if __name__ == "__main__":
    main()
