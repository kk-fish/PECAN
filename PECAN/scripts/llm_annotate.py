# -*- coding: utf-8 -*-
"""Structured policy annotation with Qwen2.5-7B-Instruct on GPU (slurm)."""
import json, os, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = "data"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
os.environ.setdefault("HF_HOME", "models/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

INDUSTRIES = ["电子信息", "电子器件", "机械行业", "汽车制造", "生物制药",
              "医疗器械", "电器行业", "发电设备", "飞机制造", "仪器仪表",
              "环保行业", "化工行业", "船舶制造"]
TOOLS = ["财政补贴", "税收优惠", "金融支持", "政府采购", "人才支持",
         "技术研发支持", "基础设施建设", "标准与规范", "市场准入", "试点示范", "监管约束"]

PROMPT = """你是产业政策分析专家。阅读下面的政策文本，输出一个JSON对象（只输出JSON，不要其他文字），字段：
- "target_industries": 该政策明确支持或涉及的行业，从以下列表中选择（可多选，无关则空数组）：{inds}
- "policy_tools": 政策使用的政策工具，从以下列表中选择（可多选）：{tools}
- "binding_strength": 政策约束力强度，整数1-5（1=倡导性愿景，3=指导性意见，5=强制性法规）
- "innovation_relevance": 政策与企业技术创新的相关程度，小数0到1
- "supply_side": 供给型政策工具占比，小数0到1
- "demand_side": 需求型政策工具占比，小数0到1
- "environment_side": 环境型政策工具占比，小数0到1

政策标题：{title}
政策正文（节选）：
{text}
"""

def extract_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    policies = [json.loads(l) for l in open(f"{DATA}/policies.jsonl")]
    done = set()
    out_path = f"{DATA}/policy_llm.jsonl"
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["url"])
            except Exception:
                pass
    fout = open(out_path, "a", encoding="utf-8")

    batch, metas = [], []
    def flush():
        if not batch:
            return
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3200).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=400, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        for j, meta in enumerate(metas):
            gen = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            d = extract_json(gen) or {}
            d["url"] = meta["url"]
            d["title"] = meta["title"]
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
        fout.flush()
        batch.clear(); metas.clear()

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n = 0
    for p in policies:
        if p["url"] in done:
            continue
        prompt = PROMPT.format(inds=INDUSTRIES, tools=TOOLS, title=p["title"], text=p["text"][:2500])
        batch.append(prompt); metas.append(p)
        if len(batch) >= 8:
            flush(); n += 8
            print(f"annotated {n}", flush=True)
    flush()
    fout.close()
    print("ANNOTATE_DONE", flush=True)

if __name__ == "__main__":
    main()
