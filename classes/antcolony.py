import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

from .maplogic import OSMMap

EdgeKey = Tuple[int, int, int]  # (u, v, key) for MultiDiGraph edges

@dataclass
class ColonyParams:
    n_ants: int = 80
    n_iters: int = 100

    alpha: float = 1.0    # pheromone importance
    beta: float = 3.0     # edge-cost importance
    gamma: float = 2.0    # destination importance

    evaporation: float = 0.3
    q: float = 1.0

    max_steps: int = 3000
    allow_revisit: bool = True


class AntColony:
    """
    Ant Colony Optimization for directed OSMnx graphs
    with a destination-aware heuristic.
    """

    def __init__(self, osmmap: OSMMap, params: ColonyParams, seed: int = 0):
        self.osmmap = osmmap
        self.p = params
        self.rng = np.random.default_rng(seed)

        # pheromone per edge
        self.tau: Dict[EdgeKey, float] = {
            (u, v, k): 1.0 for u, v, k in self.osmmap.graph.edges(keys=True)
        }

        # filled at solve-time
        self.dist_to_target: Dict[int, float] = {}

    # --------------------------------------------------
    # Heuristics
    # --------------------------------------------------

    def edge_heuristic(self, e: EdgeKey) -> float:
        """η_cost = 1 / cost"""
        c = self.osmmap.edge_cost(e)
        if math.isinf(c):
            return 0.0
        return 1.0 / c

    def goal_heuristic(self, v: int) -> float:
        """η_goal = 1 / distance_to_target"""
        d = self.dist_to_target.get(v, float("inf"))
        return 1.0 / (d + 1.0)

    # --------------------------------------------------
    # Edge selection
    # --------------------------------------------------

    def choose_edge(self, u: int, visited: set) -> Optional[EdgeKey]:
        candidates = self.osmmap.outgoing_edges(u)
        if not candidates:
            return None

        weights = []
        edges = []

        for e in candidates:
            _, v, _ = e

            if not self.p.allow_revisit and v in visited:
                continue

            eta_cost = self.edge_heuristic(e)
            if eta_cost <= 0:
                continue

            eta_goal = self.goal_heuristic(v)
            tau = self.tau[e]

            w = (
                (tau ** self.p.alpha)
                * (eta_cost ** self.p.beta)
                * (eta_goal ** self.p.gamma)
            )

            if w > 0:
                edges.append(e)
                weights.append(w)

        if not edges:
            return None

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        return edges[self.rng.choice(len(edges), p=weights)]

    # --------------------------------------------------
    # Path construction
    # --------------------------------------------------

    def construct_path(self, source: int, target: int) -> Optional[List[EdgeKey]]:
        u = source
        visited = {u}
        path = []

        for _ in range(self.p.max_steps):
            if u == target:
                return path

            e = self.choose_edge(u, visited)
            if e is None:
                return None

            path.append(e)
            _, v, _ = e
            u = v
            visited.add(u)

        return None

    # --------------------------------------------------
    # Main solve loop
    # --------------------------------------------------

    def solve(self, source: int, target: int):
        rev = self.osmmap.graph.reverse(copy=False)

        # Use NetworkX’s default handling for MultiDiGraph weights:
        # If weight is a string, it takes the MIN over parallel edges.
        # So we need a per-edge attribute that is finite.
        # base_cost is per-edge dict in OSMnx, so OK if finite.
        self.dist_to_target = nx.single_source_dijkstra_path_length(rev, target, weight="base_cost")

        best_path = None
        best_cost = float("inf")

        for _ in range(self.p.n_iters):
            solutions = []

            for _ in range(self.p.n_ants):
                path = self.construct_path(source, target)
                if path is None:
                    continue

                cost = self.osmmap.path_cost(path)
                if math.isinf(cost):
                    continue

                solutions.append((path, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            success = len(solutions)
            print("solutions this iter:", success)

            # Evaporation
            for e in self.tau:
                self.tau[e] = max(1e-12, (1 - self.p.evaporation) * self.tau[e])

            # Deposit pheromone
            for path, cost in solutions:
                delta = self.p.q / cost
                for e in path:
                    self.tau[e] += delta

        return best_path, best_cost