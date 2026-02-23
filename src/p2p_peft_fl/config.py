from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    # data/model
    model: str = "bert"              # bert | albert | roberta | <hf_id>
    dataset: str = "sst2"            # sst2 | ag_news | imdb
    peft: str = "lora"               # lora | wlora | dlora

    # p2p
    nodes: int = 3
    topology: str = "ring"           # ring | random | full | grid
    protocol: str = "gossip"         # gossip | epidemic
    degree: int = 2                  # for random topology

    # training
    rounds: int = 3
    local_steps: int = 30
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_length: int = 256

    # non-iid partition
    non_iid_alpha: float = 0.5       # Dirichlet alpha (smaller -> more skew)

    # compute / reproducibility
    device: str = "auto"             # auto | cpu | cuda
    seed: int = 123
    fp16: bool = False

    # dataset limits (for fast runs)
    max_train_samples: int = 2000
    max_eval_samples: int = 1000

    # communication / aggregation
    comm_topk: float = 1.0           # keep ratio in [0,1]; 1.0 = no sparsify
    agg: str = "mean"                # mean | wlora | dlora (auto-set from peft)
    ema_beta: float = 0.6            # for dlora
    wlora_attn_mult: float = 1.5     # for wlora (attention keys multiplier)
    wlora_mlp_mult: float = 1.0      # for wlora (mlp keys multiplier)

    # logging
    run_dir: Optional[str] = None
    log_every: int = 10              # local steps
