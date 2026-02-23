from __future__ import annotations
import argparse
import os
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from p2p_peft_fl.config import ExperimentConfig
from p2p_peft_fl.utils.seed import seed_everything
from p2p_peft_fl.utils.logging import RunLogger
from p2p_peft_fl.data.datasets import load_text_dataset
from p2p_peft_fl.data.partition import dirichlet_partition
from p2p_peft_fl.models.factory import build_model, resolve_model_id
from p2p_peft_fl.peft.factory import PeftSpec, attach_lora
from p2p_peft_fl.p2p.topology import build_topology
from p2p_peft_fl.p2p.protocols import route_messages
from p2p_peft_fl.training.node import Node
from p2p_peft_fl.training.evaluate import eval_loop
from p2p_peft_fl.training.aggregation import (
    sparsify_topk, mean_aggregate, wlora_aggregate, dlora_aggregate
)

def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="P2P Federated PEFT (LoRA) simulator")
    # data/model
    p.add_argument("--model", type=str, default=ExperimentConfig.model)
    p.add_argument("--dataset", type=str, default=ExperimentConfig.dataset)
    p.add_argument("--peft", type=str, default=ExperimentConfig.peft, choices=["lora", "wlora", "dlora"])

    # p2p
    p.add_argument("--nodes", type=int, default=ExperimentConfig.nodes)
    p.add_argument("--topology", type=str, default=ExperimentConfig.topology, choices=["ring", "random", "full", "grid"])
    p.add_argument("--protocol", type=str, default=ExperimentConfig.protocol, choices=["gossip", "epidemic"])
    p.add_argument("--degree", type=int, default=ExperimentConfig.degree)

    # training
    p.add_argument("--rounds", type=int, default=ExperimentConfig.rounds)
    p.add_argument("--local_steps", type=int, default=ExperimentConfig.local_steps)
    p.add_argument("--batch_size", type=int, default=ExperimentConfig.batch_size)
    p.add_argument("--lr", type=float, default=ExperimentConfig.lr)
    p.add_argument("--weight_decay", type=float, default=ExperimentConfig.weight_decay)
    p.add_argument("--max_length", type=int, default=ExperimentConfig.max_length)

    # non-iid
    p.add_argument("--non_iid_alpha", type=float, default=ExperimentConfig.non_iid_alpha)

    # compute
    p.add_argument("--device", type=str, default=ExperimentConfig.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    p.add_argument("--fp16", action="store_true")

    # dataset limits
    p.add_argument("--max_train_samples", type=int, default=ExperimentConfig.max_train_samples)
    p.add_argument("--max_eval_samples", type=int, default=ExperimentConfig.max_eval_samples)

    # comm/agg
    p.add_argument("--comm_topk", type=float, default=ExperimentConfig.comm_topk)
    p.add_argument("--ema_beta", type=float, default=ExperimentConfig.ema_beta)
    p.add_argument("--wlora_attn_mult", type=float, default=ExperimentConfig.wlora_attn_mult)
    p.add_argument("--wlora_mlp_mult", type=float, default=ExperimentConfig.wlora_mlp_mult)

    # logging
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--log_every", type=int, default=ExperimentConfig.log_every)

    a = p.parse_args()
    cfg = ExperimentConfig(**vars(a))
    cfg.agg = cfg.peft  # align naming
    return cfg

def pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    device = pick_device(cfg.device)
    if cfg.fp16 and device.type != "cuda":
        print("[warn] fp16 requested but CUDA not available -> disabling fp16")
        cfg.fp16 = False

    model_id_preview = resolve_model_id(cfg.model)
    print(f"[cfg] model={cfg.model}({model_id_preview}) dataset={cfg.dataset} peft={cfg.peft} "
          f"nodes={cfg.nodes} topology={cfg.topology} protocol={cfg.protocol} device={device}")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.run_dir or os.path.join("runs", run_id)
    logger = RunLogger(run_dir, cfg)

    # Load dataset + tokenizer
    tok_train, tok_test, tokenizer, num_labels = load_text_dataset(
        dataset=cfg.dataset,
        model_id=model_id_preview,
        max_length=cfg.max_length,
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )

    # Partition train set across nodes (non-IID label skew)
    parts = dirichlet_partition(tok_train["labels"], n_clients=cfg.nodes, alpha=cfg.non_iid_alpha, seed=cfg.seed)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Build nodes
    nodes: List[Node] = []
    for i in range(cfg.nodes):
        base_model, model_id = build_model(cfg.model, num_labels=num_labels)
        peft_model = attach_lora(base_model, PeftSpec())
        # subset for this node
        client_ds = tok_train.select(parts[i].tolist())
        train_loader = DataLoader(client_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
        eval_loader = DataLoader(tok_test, batch_size=cfg.batch_size, shuffle=False, collate_fn=collator)
        nodes.append(Node(
            node_id=i,
            model=peft_model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fp16=cfg.fp16,
        ))
        logger.event("node_init", node=i, train_size=len(client_ds))

    # Topology
    adj = build_topology(cfg.topology, n=cfg.nodes, degree=cfg.degree, seed=cfg.seed)
    logger.event("topology", adj=adj)

    # State for dlora EMA
    ema_state: Dict[int, Dict[str, torch.Tensor] | None] = {i: None for i in range(cfg.nodes)}

    # Main rounds
    for r in range(cfg.rounds):
        print(f"\n[round {r:03d}] local train ...")
        logger.event("round_start", round=r)

        # Local train
        for nd in nodes:
            metrics = nd.local_train(cfg.local_steps, log_every=cfg.log_every)
            logger.event("local_train", round=r, node=nd.node_id, **metrics)

        # Export adapters (optionally sparsify for communication)
        exported = {nd.node_id: sparsify_topk(nd.export_adapter(), cfg.comm_topk) for nd in nodes}

        # Route messages
        msgs = route_messages(cfg.protocol, adj, exported, seed=cfg.seed + r)
        logger.event("messages", round=r, count=len(msgs))

        # Collect inbox per node
        inbox: Dict[int, List[Dict[str, torch.Tensor]]] = {i: [exported[i]] for i in range(cfg.nodes)}  # include self
        for m in msgs:
            inbox[m.dst].append(m.payload)

        # Aggregate + import
        for nd in nodes:
            recv = inbox[nd.node_id]
            mean_state = mean_aggregate(recv)
            if cfg.agg == "mean" or cfg.agg == "lora":
                new_state = mean_state
            elif cfg.agg == "wlora":
                new_state = wlora_aggregate(recv, attn_mult=cfg.wlora_attn_mult, mlp_mult=cfg.wlora_mlp_mult)
            elif cfg.agg == "dlora":
                new_state = dlora_aggregate(ema_state[nd.node_id], mean_state, beta=cfg.ema_beta)
                ema_state[nd.node_id] = {k: v.clone() for k, v in new_state.items()}
            else:
                raise ValueError(f"Unknown agg={cfg.agg}")

            nd.import_adapter(new_state)

        # Evaluate (global test set on each node model)
        print(f"[round {r:03d}] eval ...")
        for nd in nodes:
            ev = eval_loop(nd.model, nd.eval_loader, device=device)
            logger.event("eval", round=r, node=nd.node_id, **ev)
            print(f"  node={nd.node_id} acc={ev['acc']:.4f} loss={ev['loss']:.4f}")

        logger.event("round_end", round=r)

    logger.close()
    print(f"\n[done] run_dir={run_dir}")

if __name__ == "__main__":
    main()
