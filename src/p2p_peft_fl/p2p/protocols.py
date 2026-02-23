from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import torch

AdapterState = Dict[str, torch.Tensor]

@dataclass
class Message:
    src: int
    dst: int
    payload: AdapterState

def gossip_round(adj: Dict[int, List[int]], states: Dict[int, AdapterState], seed: int) -> List[Message]:
    """Each node pushes to 1 random neighbor."""
    rng = random.Random(seed)
    msgs: List[Message] = []
    for i, neigh in adj.items():
        if not neigh: 
            continue
        j = rng.choice(neigh)
        msgs.append(Message(src=i, dst=j, payload=states[i]))
    return msgs

def epidemic_round(adj: Dict[int, List[int]], states: Dict[int, AdapterState]) -> List[Message]:
    """Each node pushes to all neighbors."""
    msgs: List[Message] = []
    for i, neigh in adj.items():
        for j in neigh:
            msgs.append(Message(src=i, dst=j, payload=states[i]))
    return msgs

def route_messages(protocol: str, adj: Dict[int, List[int]], states: Dict[int, AdapterState], seed: int) -> List[Message]:
    p = protocol.lower()
    if p == "gossip":
        return gossip_round(adj, states, seed)
    if p == "epidemic":
        return epidemic_round(adj, states)
    raise ValueError(f"Unknown protocol={protocol}. Options: gossip|epidemic")
