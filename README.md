# p2p-peft-fl — Gossip / Epidemic P2P Federated PEFT for Text Classification

[![CI](https://github.com/JavadDogani/P2P-decentralized-Federated-Fine-Tuning-of-LLM/actions/workflows/ci.yml/badge.svg)](https://github.com/JavadDogani/P2P-decentralized-Federated-Fine-Tuning-of-LLM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

A **single-process simulator** for **peer-to-peer (serverless) federated learning** that:
- trains **Transformer encoders** (BERT / RoBERTa, or any Hugging Face model id or local checkpoint for which PEFT can infer LoRA target modules; see [Known limitations](#known-limitations) for ALBERT) on **AG News / SST-2 / IMDB**
- uses **PEFT adapters** (LoRA) and **communicates only adapter weights** (the LoRA matrices plus the trainable classification head)
- supports **P2P topologies** (ring / random / fully-connected / grid)
- supports **P2P mixing protocols**:
  - **gossip**: push to 1 random neighbor per round
  - **epidemic**: push to all neighbors per round
- supports **adapter aggregation** policies (selected with `--peft`):
  - `mean` (standard, `--peft lora`): element-wise average of the node's own adapter and every adapter it received
  - `wlora` (LoRA + *layer-wise weighting* by key pattern): the same average, then each tensor is multiplied by `--wlora_attn_mult` (default 1.5) if its name matches an attention pattern, by `--wlora_mlp_mult` (default 1.0) if it matches an MLP/dense pattern, and left unchanged otherwise
  - `dlora` (LoRA + *dynamic mixing* via EMA across rounds): `beta * previous_aggregate + (1 - beta) * average`, with `beta = --ema_beta` (default 0.6); the first round uses the plain average
- splits the training set across nodes with **Dirichlet label skew** (non-IID)

> Note: `wlora` / `dlora` here are **system-level variants** (weighted + dynamic adapter mixing) that are easy to extend into a paper-grade baseline.
> The PEFT adapter itself is LoRA in all modes (r=8, alpha=16, dropout=0.05 on PEFT's default target modules, e.g. the attention query/value projections for BERT/RoBERTa), so shapes stay compatible across peers.

---

## Quick start

### 1) Install
```bash
git clone https://github.com/JavadDogani/P2P-decentralized-Federated-Fine-Tuning-of-LLM.git
cd P2P-decentralized-Federated-Fine-Tuning-of-LLM
python -m venv .venv
source .venv/bin/activate
pip install -U pip
# optional: CPU-only PyTorch wheel (much smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"   # package + dependencies + pytest/ruff
```
This installs the `p2p-peft-fl` command (equivalent to `python -m p2p_peft_fl.cli`).

### 2) Smoke run (tiny model)
```bash
p2p-peft-fl \
  --model google/bert_uncased_L-2_H-128_A-2 \
  --dataset sst2 \
  --nodes 4 \
  --topology ring \
  --protocol gossip \
  --rounds 2 \
  --local_steps 10 \
  --max_train_samples 500 \
  --max_eval_samples 200
```
Downloads a 2-layer BERT (about 18 MB) and SST-2 on first use; it finished in about 12 seconds on CPU in our test.

### 3) Run a small demo (CPU)
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
This fine-tunes `bert-base-uncased` and took 2.5–4 minutes on a shared CPU-only machine in our tests. It is a smoke run sized to finish quickly, not to train a useful model: increase `--rounds` and `--local_steps` for real experiments. For SST-2 the GLUE validation split (872 sentences) serves as the test set.

### 4) View logs
Each run prints per-round, per-node accuracy/loss to the console and writes a JSONL event log to `runs/<run_id>/events.jsonl` (or `--run_dir`). Event types: `run_start` (with the full config), `node_init`, `topology`, `round_start`, `local_train`, `messages`, `eval`, `round_end`, `run_end`.

---

## Key CLI options (all have defaults)

- `--model`: `bert | albert | roberta | distilbert | <hf_model_id or local path>` (default `bert`)
- `--dataset`: `ag_news | sst2 | imdb` (default `sst2`)
- `--peft`: `lora | wlora | dlora` — selects the aggregation policy (`lora` = plain `mean`)
- `--topology`: `ring | random | full | grid`; `--degree`: minimum degree for `random` (default 2)
- `--protocol`: `gossip | epidemic`
- `--non_iid_alpha`: Dirichlet α for label-skew partitioning (smaller = more non-IID; default 0.5)
- `--comm_topk`: optional sparsification ratio for sent adapter tensors, applied per tensor (e.g., `0.2` keeps the top-20% abs values; default `1.0` = no sparsification)
- `--ema_beta` (`dlora`), `--wlora_attn_mult` / `--wlora_mlp_mult` (`wlora`)
- `--rounds`, `--local_steps`, `--batch_size`, `--lr`, `--max_length`, `--seed`, `--device`, `--fp16` (CUDA only)

Run `p2p-peft-fl --help` for the full list.

## What happens in a round
1. Every node runs `--local_steps` AdamW steps on its own data shard (only the LoRA weights and the classification head are trainable).
2. Every node exports its adapter state (optionally top-k sparsified with `--comm_topk`).
3. Messages are routed over the topology by the protocol (gossip: one random neighbor; epidemic: all neighbors).
4. Every node aggregates its own exported adapter with the ones it received, using the selected policy, and loads the result.
5. Every node is evaluated on the same held-out test split.

---

## Project layout
```
src/p2p_peft_fl/
  cli.py                 # entrypoint + argparse; run_experiment() runs a full simulation
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
    node.py              # Node: local train + adapter export/import
    aggregation.py       # adapter aggregation (mean/wlora/dlora) + top-k sparsification
    evaluate.py          # metrics
  utils/
    seed.py              # reproducibility
    logging.py           # JSONL run logger
tests/                   # offline pytest suite
```

---

## Testing
```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
The suite runs fully offline in a few seconds on CPU: end-to-end runs use a tiny randomly initialised BERT saved to a temporary directory and synthetic sentences, so nothing is downloaded. It covers topology construction (degree, symmetry, connectivity), gossip vs epidemic message counts, the aggregation rules and top-k sparsification, Dirichlet partitioning (every sample assigned exactly once, skew increasing as α decreases), tokenization, adapter export/import, argument parsing, and complete simulation runs (including consensus under fully-connected epidemic averaging and seed reproducibility). The known limitations below are recorded as strict `xfail` tests. CI runs the suite on Python 3.10, 3.11 and 3.12 with the CPU-only PyTorch wheel.

## Known limitations
- `--model albert` and `--model distilbert` currently fail when the LoRA adapter is attached: `PeftSpec.target_modules` is `None`, and PEFT has no default target modules for these architectures. BERT and RoBERTa work.
- A 2-node `ring` lists the single neighbor twice, so `epidemic` delivers each message twice.
- The `random` topology only enforces a minimum degree; it is not guaranteed to be connected (e.g. 9 nodes, degree 2, seed 0 gives two components).
- With a very small `--non_iid_alpha` and many nodes, a node can receive no training samples, which currently stops the run with an error.
- `wlora` rescales rather than re-weights peers: with `--wlora_attn_mult` ≠ 1, the attention LoRA factors are multiplied by that value at every round, so the effect compounds over rounds.

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

---

## Citing
If you use this code, please cite it via the metadata in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button generates APA and BibTeX from it):
```bibtex
@software{dogani_p2p_peft_fl,
  author  = {Dogani, Javad},
  title   = {p2p-peft-fl: Gossip / Epidemic P2P Federated PEFT for Text Classification},
  version = {0.1.0},
  url     = {https://github.com/JavadDogani/P2P-decentralized-Federated-Fine-Tuning-of-LLM}
}
```

## License
MIT — see [LICENSE](LICENSE).
