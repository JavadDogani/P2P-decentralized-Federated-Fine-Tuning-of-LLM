from __future__ import annotations

import pytest
import torch

from p2p_peft_fl.p2p.protocols import epidemic_round, gossip_round, route_messages
from p2p_peft_fl.p2p.topology import build_topology

TOPOLOGIES = [("ring", 5), ("full", 4), ("grid", 7), ("random", 8)]


def states_for(n: int):
    return {i: {"w": torch.full((2, 2), float(i))} for i in range(n)}


@pytest.mark.parametrize("kind,n", TOPOLOGIES)
def test_gossip_sends_one_message_per_node_to_a_neighbour(kind, n):
    adj = build_topology(kind, n=n, degree=2, seed=0)
    msgs = gossip_round(adj, states_for(n), seed=3)
    assert len(msgs) == n
    assert sorted(m.src for m in msgs) == list(range(n))
    for m in msgs:
        assert m.dst in adj[m.src]


@pytest.mark.parametrize("kind,n", TOPOLOGIES)
def test_epidemic_sends_to_every_neighbour(kind, n):
    adj = build_topology(kind, n=n, degree=2, seed=0)
    msgs = epidemic_round(adj, states_for(n))
    assert len(msgs) == sum(len(neigh) for neigh in adj.values())
    assert sorted((m.src, m.dst) for m in msgs) == sorted((i, j) for i, neigh in adj.items() for j in neigh)


@pytest.mark.parametrize("n", [3, 6])
def test_message_counts_for_ring_and_full(n):
    ring = build_topology("ring", n=n, degree=2, seed=0)
    full = build_topology("full", n=n, degree=2, seed=0)
    assert len(route_messages("gossip", ring, states_for(n), seed=0)) == n
    assert len(route_messages("epidemic", ring, states_for(n), seed=0)) == 2 * n
    assert len(route_messages("gossip", full, states_for(n), seed=0)) == n
    assert len(route_messages("epidemic", full, states_for(n), seed=0)) == n * (n - 1)


def test_payload_is_the_senders_state():
    n = 4
    states = states_for(n)
    adj = build_topology("full", n=n, degree=2, seed=0)
    for msgs in (gossip_round(adj, states, seed=0), epidemic_round(adj, states)):
        for m in msgs:
            assert m.payload is states[m.src]


def test_gossip_is_deterministic_per_seed_and_varies_across_seeds():
    n = 10
    adj = build_topology("full", n=n, degree=2, seed=0)
    pairs = lambda seed: [(m.src, m.dst) for m in gossip_round(adj, states_for(n), seed=seed)]  # noqa: E731
    assert pairs(5) == pairs(5)
    assert len({tuple(pairs(s)) for s in range(5)}) > 1


def test_isolated_nodes_send_nothing():
    adj = {0: [], 1: [2], 2: [1]}
    assert len(gossip_round(adj, states_for(3), seed=0)) == 2
    assert len(epidemic_round(adj, states_for(3))) == 2


def test_protocol_name_is_case_insensitive_and_validated():
    adj = build_topology("ring", n=3, degree=2, seed=0)
    assert len(route_messages("EPIDEMIC", adj, states_for(3), seed=0)) == 6
    with pytest.raises(ValueError, match="Unknown protocol"):
        route_messages("flooding", adj, states_for(3), seed=0)
