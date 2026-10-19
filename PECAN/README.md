# PECAN: policy-enterprise coupling attention network

Code and data for the paper *Language models and graph attention map the coupling between industrial policy texts and corporate R&D in individual firms* (Bohan Fei, Columbia University).

PECAN links 690 Chinese State Council and ministry policy documents to 1,003 A-share listed firms. Qwen2.5-7B-Instruct extracts a structured profile of each document, BGE-large-zh-v1.5 embeds policies and firm business descriptions, yearly heterogeneous graphs connect policies to firms, and a graph attention channel fused with an XGBoost channel forecasts each firm's R&D intensity and R&D expansion in the following year. Models are trained on feature years 2019 to 2022, validated on 2023, and tested on 2024 features (fiscal 2025 outcomes).

## Repository layout

```
scripts/                 data collection, annotation, graph building, training, statistics
  crawl_policies.py      crawl policy documents from the gov.cn policy library
  collect_firms.py       firm universe (Sina Finance), business profiles (Tonghuashun), income statements (Eastmoney)
  llm_annotate.py        structured policy profiles with Qwen2.5-7B-Instruct
  embed_texts.py         BGE-large-zh-v1.5 embeddings of policies and firms
  build_graph.py         yearly policy-firm graphs (data/graphs.json)
  make_policy_meta.py    lookup table of policy titles and LLM profiles (data/policy_meta.json)
  train.py               PECAN, ablations, and baselines, five seeds (Tables 2 to 4)
  train_case.py          same models, seed 0, with firm-level predictions and saved weights (Figure 8)
  stats_tests.py         means, SDs, Welch's t tests, fusion weights, and firm-level shares quoted in the paper
figures/
  plot_figures.py        Figures 3, 4, and 7
  plot_case.py           Figure 5
  plot_case2.py          Figure 6
  plot_case_pred.py      Figure 8
  tikz/                  TikZ sources of Figure 1, Figure 2, and the graphical abstract
data/                    processed data used in the paper (see below)
results/                 saved results behind the tables and figures
```

## Data

| File | Content | Source |
|---|---|---|
| `policies.jsonl` | 690 policy documents (title, URL, date, collection, full text), 2016 to 2026 | State Council policy library, https://www.gov.cn/zhengce/ |
| `policy_llm.jsonl` | LLM profile of each document (target industries, tools, binding strength, innovation relevance, supply/demand/environment shares) | output of `llm_annotate.py` |
| `policy_meta.json` | url → title, date, and LLM profile | output of `make_policy_meta.py` |
| `policy_emb.npy`, `policy_order.json` | 1,024-dimensional policy embeddings and their row order | output of `embed_texts.py` |
| `universe.csv` | 1,003 firms with stock code and industry (13 sectors) | Sina Finance industry classification |
| `firm_text.jsonl` | main business, product types, and registered business scope | Tonghuashun (10jqka) |
| `firm_fin.jsonl` | operating revenue, R&D expense, and net profit, fiscal years 2018 to 2025 | Eastmoney annual income statements (via AKShare) |
| `firm_emb.npy`, `firm_order.json`, `graph_firm_emb.npy` | firm embeddings and row order | output of `embed_texts.py` / `build_graph.py` |
| `graphs.json` | yearly graphs (feature years 2019 to 2024): policy nodes, policy edges with attributes, peer edges, firm features, labels | output of `build_graph.py` |

The raw texts and financial figures come from the public websites listed above and remain subject to their terms of use.

`results/` contains `results_final.csv` (five seeds, all models; Tables 2 to 4 and Figures 3 and 4), `preds_dual_final.json` (Figure 7), `attention_dual_final.json` (policy-firm attention on the 2024 test graph; Figures 5 and 6), and `case/` (seed 0 firm-level predictions of every model; Figure 8).

## Environment

Python 3.10 with the packages in `requirements.txt` (tested with PyTorch 2.7.1 + CUDA 12.6 and PyTorch Geometric 2.8.0). The LLM annotation and the embeddings need a GPU; we used one NVIDIA A800 (40 GB). Training a model takes a few minutes on one GPU.

```bash
pip install -r requirements.txt
```

## Reproducing the paper

Run every command from the repository root.

```bash
# 1. data (optional: the processed files are already in data/)
python scripts/crawl_policies.py
python scripts/collect_firms.py
python scripts/llm_annotate.py          # GPU, downloads Qwen/Qwen2.5-7B-Instruct
python scripts/embed_texts.py           # GPU, downloads BAAI/bge-large-zh-v1.5
python scripts/build_graph.py
python scripts/make_policy_meta.py

# 2. models: five seeds of every model in Tables 2 to 4 (GPU)
python scripts/train.py --models pecan_dual,pecan_dual_no_llm,pecan_dual_no_policy,pecan_dual_no_semantic,pecan_dual_no_ff,pecan,mlp,gcn,gat,sage,xgboost,xgb_nopolicy --horizon 1 --tag _final

# 3. firm-level predictions for the case studies (seed 0, GPU)
python scripts/train_case.py --models pecan_dual,pecan,mlp,gcn,gat,sage,xgboost,xgb_nopolicy,pecan_dual_no_ff,pecan_dual_no_policy --seeds 1 --horizon 1 --tag _case

# 4. statistics and figures (CPU)
python scripts/stats_tests.py
python figures/plot_figures.py
python figures/plot_case.py
python figures/plot_case2.py
python figures/plot_case_pred.py
cd figures/tikz && pdflatex pecan_framework.tex && pdflatex pecan_arch.tex && pdflatex pecan_graphical_abstract.tex
```

Model names: `pecan_dual` is the full model (neural and tree channels); `pecan` is the neural channel only; `pecan_dual_no_llm`, `_no_policy`, `_no_semantic`, and `_no_ff` remove the LLM structured features, the policy branch, the semantic policy edges, and peer propagation; `xgboost` and `xgb_nopolicy` are XGBoost with and without the policy exposure features.

Graph attention layers use nondeterministic GPU operations, so a rerun reproduces the reported metrics to within about 0.001 RMSE and 0.005 AUC rather than exactly.

## Contact

Bohan Fei, tf2667@columbia.edu
