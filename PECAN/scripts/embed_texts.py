# -*- coding: utf-8 -*-
"""Encode policy and firm texts with bge-large-zh-v1.5 on GPU."""
import json, os
import numpy as np

os.environ.setdefault("HF_HOME", "models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from sentence_transformers import SentenceTransformer

DATA = "data"

def main():
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device="cuda")
    policies = [json.loads(l) for l in open(f"{DATA}/policies.jsonl")]
    firms = [json.loads(l) for l in open(f"{DATA}/firm_text.jsonl")]

    p_texts = [p["title"] + "。" + p["text"][:1500] for p in policies]
    f_texts = [f"{f['name']}，所属行业：{f['industry']}。主营业务：{f['main_business']} 产品：{f['product_type']} 经营范围：{f['scope'][:500]}" for f in firms]

    p_emb = model.encode(p_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    f_emb = model.encode(f_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

    np.save(f"{DATA}/policy_emb.npy", p_emb)
    np.save(f"{DATA}/firm_emb.npy", f_emb)
    json.dump([p["url"] for p in policies], open(f"{DATA}/policy_order.json", "w"))
    json.dump([f["code"] for f in firms], open(f"{DATA}/firm_order.json", "w"))
    print("EMBED_DONE", p_emb.shape, f_emb.shape, flush=True)

if __name__ == "__main__":
    main()
