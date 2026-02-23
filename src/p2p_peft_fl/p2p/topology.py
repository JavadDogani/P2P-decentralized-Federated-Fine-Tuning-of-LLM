from __future__ import annotations
from typing import Dict, List
import random

def build_topology(kind: str, n: int, degree: int, seed: int) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    kind = kind.lower()
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}

    if n <= 1:
        return adj

    if kind == "ring":
        for i in range(n):
            adj[i] = [ (i-1) % n, (i+1) % n ]
        return adj

    if kind == "full":
        for i in range(n):
            adj[i] = [j for j in range(n) if j != i]
        return adj

    if kind == "grid":
        # 2D grid as close to square as possible
        import math
        side = int(math.ceil(math.sqrt(n)))
        def nid(r, c): return r * side + c
        for r in range(side):
            for c in range(side):
                i = nid(r, c)
                if i >= n: 
                    continue
                neigh = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    j = nid(rr, cc)
                    if 0 <= rr < side and 0 <= cc < side and j < n:
                        neigh.append(j)
                adj[i] = neigh
        return adj

    if kind == "random":
        # simple undirected random graph with approx degree
        deg = max(1, min(degree, n-1))
        edges = set()
        for i in range(n):
            while len([1 for (a,b) in edges if a == i or b == i]) < deg:
                j = rng.randrange(n)
                if j == i: 
                    continue
                a, b = min(i,j), max(i,j)
                edges.add((a,b))
        for a,b in edges:
            adj[a].append(b)
            adj[b].append(a)
        # ensure no empty neighbors
        for i in range(n):
            if not adj[i]:
                j = (i+1) % n
                adj[i] = [j]
                adj[j].append(i)
        return adj

    raise ValueError(f"Unknown topology={kind}. Options: ring|random|full|grid")
