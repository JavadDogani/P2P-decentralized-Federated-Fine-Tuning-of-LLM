from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import torch

AdapterState = Dict[str, torch.Tensor]

def sparsify_topk(state: AdapterState, keep_ratio: float) -> AdapterState:
    """Top-k sparsification per tensor (by abs value). keep_ratio=1 -> no-op."""
    if keep_ratio >= 1.0:
        return state
    out: AdapterState = {}
    for k, v in state.items():
        flat = v.view(-1)
        n = flat.numel()
        kk = max(1, int(keep_ratio * n))
        # threshold
        if kk >= n:
            out[k] = v
            continue
        vals = flat.abs()
        thresh = torch.topk(vals, kk, largest=True, sorted=False).values.min()
        mask = (vals >= thresh).to(v.dtype)
        out[k] = (flat * mask).view_as(v)
    return out

def mean_aggregate(states: List[AdapterState]) -> AdapterState:
    if not states:
        raise ValueError("No states to aggregate")
    keys = states[0].keys()
    out: AdapterState = {}
    for k in keys:
        stack = torch.stack([s[k] for s in states], dim=0)
        out[k] = stack.mean(dim=0)
    return out

def wlora_aggregate(states: List[AdapterState], attn_mult: float, mlp_mult: float) -> AdapterState:
    """Layer-wise weighted mean by key-pattern. Strong baseline for adapter mixing."""
    if not states:
        raise ValueError("No states to aggregate")
    keys = states[0].keys()
    out: AdapterState = {}
    for k in keys:
        if any(tok in k.lower() for tok in ["attn", "attention", "query", "key", "value"]):
            w = attn_mult
        elif any(tok in k.lower() for tok in ["mlp", "ff", "dense", "intermediate", "output"]):
            w = mlp_mult
        else:
            w = 1.0
        stack = torch.stack([s[k] for s in states], dim=0)
        out[k] = (stack.mean(dim=0) * w)
    return out

def dlora_aggregate(prev: AdapterState | None, new_mean: AdapterState, beta: float) -> AdapterState:
    """EMA mixing across rounds (dynamic). prev=None -> new_mean."""
    if prev is None:
        return new_mean
    out: AdapterState = {}
    for k in new_mean.keys():
        out[k] = beta * prev[k] + (1.0 - beta) * new_mean[k]
    return out
