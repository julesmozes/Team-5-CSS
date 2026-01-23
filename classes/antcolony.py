import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

EdgeKey = Tuple[int, int, int]  # (u, v, key) for MultiDiGraph edges

@dataclass
class ACOParams:
    n_ants: int = 50
    n_iters: int = 60
    alpha: float = 1.0           # pheromone influence
    beta: float = 3.0            # heuristic influence
    evaporation: float = 0.3     # pheromone evaporation rate
    q: float = 1.0               # deposit constant
    max_steps: int = 2000        # safety against wandering
    allow_revisit: bool = False


class ACORouter:
    """
    Ant Colony Optimization on a directed MultiDiGraph.
    Pheromone is stored per edge (u,v,key).
    """

    def __init__(self, G: nx.MultiDiGraph, params: Optional[ACOParams] = None, seed: int = 0):
        self.G = G
        self.p = params or ACOParams()
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

        # Initialize pheromone on all edges
        self.tau: Dict[EdgeKey, float] = {}
        for u, v, k in self.G.edges(keys=True):
            self.tau[(u, v, k)] = 1.0

    def _heuristic(self, e: EdgeKey) -> float:
        """
        Heuristic desirability eta = 1 / cost.
        """
        c = edge_cost(self.G, e)
        if math.isinf(c):
            return 0.0
        return 1.0 / c

    def _choose_next_edge(self, u: int, visited_nodes: set) -> Optional[EdgeKey]:
        """
        Probabilistically select an outgoing edge from node u.
        Closed edges automatically get probability 0 via heuristic.
        """
        candidates = outgoing_edges(self.G, u)
        if not candidates:
            return None

        weights = []
        kept = []
        for e in candidates:
            _, v, _ = e

            if (not self.p.allow_revisit) and (v in visited_nodes):
                continue

            eta = self._heuristic(e)
            if eta <= 0.0:
                continue

            tau = self.tau.get(e, 1.0)
            w = (tau ** self.p.alpha) * (eta ** self.p.beta)
            if w > 0:
                kept.append(e)
                weights.append(w)

        if not kept:
            return None

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        idx = self.nprng.choice(len(kept), p=weights)
        return kept[int(idx)]

    def _construct_solution(self, source: int, target: int) -> Optional[List[EdgeKey]]:
        """
        Build a path as a sequence of edges.
        """
        u = source
        visited = {u}
        edges: List[EdgeKey] = []

        for _ in range(self.p.max_steps):
            if u == target:
                return edges

            e = self._choose_next_edge(u, visited)
            if e is None:
                return None

            edges.append(e)
            _, v, _ = e
            u = v
            visited.add(u)

        return None  # exceeded max_steps

    def _evaporate(self) -> None:
        rho = self.p.evaporation
        for e in list(self.tau.keys()):
            self.tau[e] = max(1e-12, (1.0 - rho) * self.tau[e])

    def _deposit(self, solutions: List[Tuple[List[EdgeKey], float]]) -> None:
        """
        Deposit pheromone inversely proportional to path cost.
        """
        for path_edges, cost in solutions:
            if cost <= 0 or math.isinf(cost):
                continue
            delta = self.p.q / cost
            for e in path_edges:
                self.tau[e] = self.tau.get(e, 1.0) + delta

    def solve(self, source: int, target: int) -> Tuple[Optional[List[EdgeKey]], float]:
        """
        Run ACO and return best edge-path and its cost.
        """
        best_path = None
        best_cost = float("inf")

        for _ in range(self.p.n_iters):
            iter_solutions: List[Tuple[List[EdgeKey], float]] = []

            for _ant in range(self.p.n_ants):
                path_edges = self._construct_solution(source, target)
                if path_edges is None:
                    continue
                c = path_cost(self.G, path_edges)
                if math.isinf(c):
                    continue

                iter_solutions.append((path_edges, c))
                if c < best_cost:
                    best_cost = c
                    best_path = path_edges

            # Update pheromone
            self._evaporate()
            # You can deposit all, or only top-k. Here: deposit all found this iter.
            self._deposit(iter_solutions)

        return best_path, best_cost
