# p2p-peft-fl — Gossip / Epidemic P2P Federated PEFT for Text Classification

A **single-process simulator** for **peer-to-peer (serverless) federated learning** that:
- trains **Transformer encoders** (BERT / ALBERT / RoBERTa) on **AG News / SST-2 / IMDB**
- uses **PEFT adapters** (LoRA) and **communicates only adapter weights**
- supports **P2P topologies** (ring / random / fully-connected / grid)
- supports **P2P mixing protocols**:
  - **gossip**: push to 1 random neighbor per round
  - **epidemic**: push to all neighbors per round
- supports **adapter aggregation** policies:
  - `mean` (standard)
  - `wlora` (LoRA + *layer-wise weighted* averaging by key pattern)
  - `dlora` (LoRA + *dynamic mixing* via EMA across rounds)

> Note: `wlora` / `dlora` here are **strong system-level variants** (weighted + dynamic adapter mixing) that are easy to extend into a paper-grade baseline.
> The PEFT adapter itself is LoRA in all modes (so shapes stay compatible across peers).

---

## Quick start

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -r requirements.txt
```

### 2) Run a small demo (CPU)
```bash
python -m p2p_peft_fl.cli \
  --model bert \
  --dataset sst2 \
  --peft lora \
  --nodes 3 \
  --topology ring \
  --protocol gossip \
  --rounds 3 \
  --local_steps 30 \
  --max_train_samples 2000 \
  --max_eval_samples 1000
```

### 3) View logs
Each run creates:
- a readable console log
- a JSONL event log under `runs/<run_id>/events.jsonl`

---

## Key CLI options (all have defaults)

- `--model`: `bert | albert | roberta | <hf_model_id>`
- `--dataset`: `ag_news | sst2 | imdb`
- `--peft`: `lora | wlora | dlora`
- `--topology`: `ring | random | full | grid`
- `--protocol`: `gossip | epidemic`
- `--non_iid_alpha`: Dirichlet α for label-skew partitioning (smaller = more non-IID)
- `--comm_topk`: optional sparsification ratio for sent adapter tensors (e.g., `0.2` keeps top-20% abs values)

---

## Project layout
```
src/p2p_peft_fl/
  cli.py                 # entrypoint + argparse
  config.py              # dataclasses for experiment configuration
  data/
    datasets.py          # HF datasets loading + tokenization
    partition.py         # Dirichlet non-IID splitting
  models/
    factory.py           # HF model factory (sequence classification)
  peft/
    factory.py           # LoRA attach + adapter (de)serialization helpers
  p2p/
    topology.py          # adjacency construction
    protocols.py         # gossip / epidemic message passing
  training/
    node.py              # Node: local train + eval
    aggregation.py       # adapter aggregation (mean/wlora/dlora)
    evaluate.py          # metrics
  utils/
    seed.py              # reproducibility
    logging.py           # JSONL run logger
```

---

## Reproducibility tips
- Fix seed with `--seed 123`
- Pin sample counts with `--max_train_samples` / `--max_eval_samples`
- Use `--device cuda` if you have GPU; otherwise keep nodes small on CPU.

---

## Extending into a “very strong” repo/paper
You can add:
- **async gossip** + staleness-aware mixing
- **compression**: quantization / error feedback / sketches
- **robust aggregation**: trimmed mean / coordinate median / Krum-like for adapters
- **privacy**: secure aggregation-compatible adapter sketching
- **overlay networks**: similarity-based neighbor selection (semantic overlays)

See `training/aggregation.py` and `p2p/protocols.py` — they’re designed as extension points.
