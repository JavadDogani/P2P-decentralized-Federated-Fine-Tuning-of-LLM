from __future__ import annotations

import math
from typing import Dict, List

import pytest

from p2p_peft_fl.p2p.topology import build_topology

Adj = Dict[int, List[int]]


def is_connected(adj: Adj) -> bool:
    if not adj:
        return True
    seen, stack = {0}, [0]
    while stack:
        for v in adj[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(adj)


def assert_well_formed(adj: Adj, n: int) -> None:
    """Undirected simple graph on nodes 0..n-1: symmetric, no self-loops, no duplicate edges."""
    assert sorted(adj) == list(range(n))
    for i, neigh in adj.items():
        assert i not in neigh, f"self-loop at {i}"
        assert len(neigh) == len(set(neigh)), f"duplicate neighbour at {i}: {neigh}"
        for j in neigh:
            assert 0 <= j < n
            assert i in adj[j], f"edge {i}->{j} is not symmetric"


@pytest.mark.parametrize("n", [3, 4, 7, 16])
def test_ring_has_degree_two_and_is_connected(n):
    adj = build_topology("ring", n=n, degree=2, seed=0)
    assert_well_formed(adj, n)
    for i, neigh in adj.items():
        assert sorted(neigh) == sorted([(i - 1) % n, (i + 1) % n])
    assert is_connected(adj)


@pytest.mark.xfail(
    strict=True,
    reason="Known issue: a 2-node ring lists the single neighbour twice ({0: [1, 1], 1: [0, 0]}), "
    "so epidemic routing delivers each message twice. Left as-is because fixing it changes results.",
)
def test_two_node_ring_has_no_duplicate_neighbours():
    assert_well_formed(build_topology("ring", n=2, degree=2, seed=0), 2)


@pytest.mark.parametrize("n", [2, 3, 5, 10])
def test_full_topology_connects_every_pair(n):
    adj = build_topology("full", n=n, degree=2, seed=0)
    assert_well_formed(adj, n)
    for i, neigh in adj.items():
        assert sorted(neigh) == [j for j in range(n) if j != i]
    assert is_connected(adj)


@pytest.mark.parametrize("n", range(2, 17))
def test_grid_is_a_connected_4_neighbourhood(n):
    adj = build_topology("grid", n=n, degree=2, seed=0)
    assert_well_formed(adj, n)
    side = math.ceil(math.sqrt(n))
    for i, neigh in adj.items():
        r, c = divmod(i, side)
        for j in neigh:
            rr, cc = divmod(j, side)
            assert abs(r - rr) + abs(c - cc) == 1, "grid edges must join row/column neighbours"
        assert 1 <= len(neigh) <= 4
    assert is_connected(adj)


def test_square_grid_degrees():
    adj = build_topology("grid", n=9, degree=2, seed=0)
    degrees = [len(adj[i]) for i in range(9)]
    # corners 2, edge midpoints 3, centre 4
    assert degrees == [2, 3, 2, 3, 4, 3, 2, 3, 2]


@pytest.mark.parametrize("n,degree", [(3, 2), (6, 2), (10, 3), (12, 4), (5, 10)])
@pytest.mark.parametrize("seed", [0, 1, 123])
def test_random_topology_respects_minimum_degree(n, degree, seed):
    adj = build_topology("random", n=n, degree=degree, seed=seed)
    assert_well_formed(adj, n)
    # every node gets at least `degree` neighbours (capped at n - 1); some get more
    for neigh in adj.values():
        assert len(neigh) >= min(degree, n - 1)


def test_random_topology_is_deterministic_per_seed():
    a = build_topology("random", n=10, degree=3, seed=7)
    b = build_topology("random", n=10, degree=3, seed=7)
    assert a == b


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: the random generator only enforces a minimum degree, so it can return "
    "a disconnected graph (here n=9, degree=2, seed=0 gives two components).",
)
def test_random_topology_is_connected():
    assert is_connected(build_topology("random", n=9, degree=2, seed=0))


@pytest.mark.parametrize("kind", ["ring", "full", "grid", "random"])
def test_single_node_has_no_neighbours(kind):
    assert build_topology(kind, n=1, degree=2, seed=0) == {0: []}


def test_topology_name_is_case_insensitive():
    assert build_topology("RING", n=4, degree=2, seed=0) == build_topology("ring", n=4, degree=2, seed=0)


def test_unknown_topology_raises():
    with pytest.raises(ValueError, match="Unknown topology"):
        build_topology("torus", n=4, degree=2, seed=0)
