from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from classes.antcolony import ACORouter, ACOParams

EdgeKey = Tuple[int, int, int]  # (u, v, key) for MultiDiGraph edges


def build_osm_graph(place: str, network_type: str = "drive", add_travel_time: bool = True) -> nx.MultiDiGraph:
    G = ox.graph_from_place(place, network_type=network_type, simplify=True)

    # Ensure we have edge lengths (usually already present)
    if not any("length" in data for *_, data in G.edges(keys=True, data=True)):
        G = ox.distance.add_edge_lengths(G)

    # Optional: infer speeds + travel times
    if add_travel_time:
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)

    # Add dynamic state fields
    for u, v, k, data in G.edges(keys=True, data=True):
        data.setdefault("open", True)

        base = float(data["travel_time"]) if add_travel_time and "travel_time" in data else float(data["length"])
        data["base_cost"] = base
        data.setdefault("cost_mult", 1.0)
        data.setdefault("cost_add", 0.0)

    return G


def edge_cost(G: nx.MultiDiGraph, e: EdgeKey) -> float:
    """
    Dynamic edge cost: closed edges become impassable (infinite cost).
    """
    u, v, k = e
    data = G.edges[u, v, k]
    if not data.get("open", True):
        return float("inf")

    base = float(data.get("base_cost", 1.0))
    mult = float(data.get("cost_mult", 1.0))
    add = float(data.get("cost_add", 0.0))
    cost = base * mult + add

    # Guard against zero/negative costs
    return max(cost, 1e-9)


def outgoing_edges(G: nx.MultiDiGraph, u: int) -> List[EdgeKey]:
    """
    Return outgoing directed edges (u -> v) as (u,v,key).
    """
    out = []
    for _, v, k in G.out_edges(u, keys=True):
        out.append((u, v, k))
    return out


def path_cost(G: nx.MultiDiGraph, path_edges: List[EdgeKey]) -> float:
    return sum(edge_cost(G, e) for e in path_edges)


def close_edge(G: nx.MultiDiGraph, u: int, v: int, key: int = 0) -> None:
    G.edges[u, v, key]["open"] = False


def open_edge(G: nx.MultiDiGraph, u: int, v: int, key: int = 0) -> None:
    G.edges[u, v, key]["open"] = True


def set_edge_multiplier(G: nx.MultiDiGraph, u: int, v: int, key: int, mult: float) -> None:
    G.edges[u, v, key]["cost_mult"] = float(mult)


def add_edge_penalty(G: nx.MultiDiGraph, u: int, v: int, key: int, penalty: float) -> None:
    G.edges[u, v, key]["cost_add"] = float(penalty)


def close_edges_by_osmid(G: nx.MultiDiGraph, osmid: int) -> int:
    """
    Convenience: close all edges whose 'osmid' matches (or contains) osmid.
    Returns number of edges affected.
    Note: OSMnx edges can have osmid as int or list.
    """
    n = 0
    for u, v, k, data in G.edges(keys=True, data=True):
        oid = data.get("osmid", None)
        if oid is None:
            continue
        if oid == osmid or (isinstance(oid, (list, tuple, set)) and osmid in oid):
            data["open"] = False
            n += 1
    return n

# ---------------------------
# Helpers for using OSMnx nodes
# ---------------------------

def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    return int(ox.distance.nearest_nodes(G, X=lon, Y=lat))


def edges_to_nodes(source: int, path_edges: List[EdgeKey]) -> List[int]:
    """
    Convert edge-list to node-list for plotting.
    """
    nodes = [source]
    for _, v, _ in path_edges:
        nodes.append(v)
    return nodes

def largest_strongly_connected_component(G):
    sccs = nx.strongly_connected_components(G)
    largest = max(sccs, key=len)
    return G.subgraph(largest).copy()

def random_node_pair(G, min_dist_m=50, max_dist_m=100, max_tries=10000):
    nodes = list(G.nodes)

    for _ in range(max_tries):
        src = random.choice(nodes)
        dst = random.choice(nodes)
        if src == dst:
            continue

        # Euclidean separation (cheap pre-filter)
        x1, y1 = G.nodes[src]["x"], G.nodes[src]["y"]
        x2, y2 = G.nodes[dst]["x"], G.nodes[dst]["y"]
        eucl = math.hypot(x1 - x2, y1 - y2)
        if eucl < min_dist_m:
            continue

        # Directed network distance (true difficulty)
        try:
            d = nx.shortest_path_length(G, src, dst, weight="base_cost")
        except nx.NetworkXNoPath:
            continue

        if d <= max_dist_m:
            return src, dst

    raise RuntimeError("Could not find suitable node pair within distance bounds")

# 1) Build graph
G = build_osm_graph("Oost, Amsterdam, Netherlands", network_type="drive", add_travel_time=True)
G = largest_strongly_connected_component(G)
G = ox.project_graph(G) 

# checks
bad = 0
for u, v, k, d in G.edges(keys=True, data=True):
    bc = d.get("base_cost", None)
    if bc is None or not math.isfinite(float(bc)):
        bad += 1
print("Edges with non-finite/missing base_cost:", bad, "out of", G.number_of_edges())


# 2) Pick start/end (example coordinates)
src, dst = random_node_pair(G)

print("crs:", G.graph.get("crs"))
print("nodes/edges:", len(G), G.number_of_edges())
print("outdeg(src):", G.out_degree(src), "indeg(dst):", G.in_degree(dst))

fig, ax = ox.plot_graph(
    G,
    node_size=0,
    edge_linewidth=0.6,
    bgcolor="white",
    show=False,
    close=False
)

ax.scatter(
    G.nodes[src]["x"],
    G.nodes[src]["y"],
    c="green",
    s=80,
    zorder=5,
    label="source",
)

ax.scatter(
    G.nodes[dst]["x"],
    G.nodes[dst]["y"],
    c="red",
    s=80,
    zorder=5,
    label="target",
)

ax.legend()
plt.show()


try:
    sp_len = nx.shortest_path_length(G, src, dst, weight="base_cost")
    print("Dijkstra base_cost:", sp_len)
except nx.NetworkXNoPath:
    print("No directed path from src to dst (unexpected if SCC worked).")

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

    def __init__(self, G: nx.MultiDiGraph, params: ColonyParams, seed: int = 0):
        self.G = G
        self.p = params
        self.rng = np.random.default_rng(seed)

        # pheromone per edge
        self.tau: Dict[EdgeKey, float] = {
            (u, v, k): 1.0 for u, v, k in G.edges(keys=True)
        }

        # filled at solve-time
        self.dist_to_target: Dict[int, float] = {}

    # --------------------------------------------------
    # Heuristics
    # --------------------------------------------------

    def edge_heuristic(self, e: EdgeKey) -> float:
        """η_cost = 1 / cost"""
        c = edge_cost(self.G, e)
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
        candidates = outgoing_edges(self.G, u)
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
        rev = self.G.reverse(copy=False)

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

                cost = path_cost(self.G, path)
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


# 3) Run ACO
colony = AntColony(
    G,
    ColonyParams(
        n_ants=80,
        n_iters=5,
        alpha=1.0,
        beta=3.0,
        gamma=4.0,
        evaporation=0.3,
        allow_revisit=True,
        max_steps=10000
    ),
    seed=42,
)

best_edges, best_cost = colony.solve(src, dst)
print("Best cost:", best_cost)


# 4) Visualize result
if best_edges:
    route_nodes = edges_to_nodes(src, best_edges)
    fig, ax = ox.plot_graph_route(G, route_nodes, route_linewidth=4, node_size=0)