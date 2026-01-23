import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

EdgeKey = Tuple[int, int, int]  # (u, v, key) for MultiDiGraph edges

class OSMMap:
    def __init__(self, place: str = "Oost, Amsterdam, Netherlands", network_type: str = "drive", add_travel_time: bool = True):
        self.graph = self.build_osm_graph(place, network_type, add_travel_time)

    
    def build_osm_graph(self, place: str, network_type: str = "drive", add_travel_time: bool = True) -> nx.MultiDiGraph:
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

        G = self.largest_strongly_connected_component(G)
        G = ox.project_graph(G) 
        return G
    
    def largest_strongly_connected_component(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        sccs = nx.strongly_connected_components(G)
        largest = max(sccs, key=len)
        return G.subgraph(largest).copy()
    
    def set_random_pair(self, min_dist_m :float = 50, max_dist_m :float = 100, max_tries : int = 10000):
        self.src, self.dst = self.random_node_pair(self.graph, min_dist_m, max_dist_m, max_tries)
        print("crs:", self.graph.graph.get("crs"))
        print("nodes/edges:", len(self.graph), self.graph.number_of_edges())
        print("outdeg(src):", self.graph.out_degree(self.src), "indeg(dst):", self.graph.in_degree(self.dst))
    
    def random_node_pair(self, G: nx.MultiDiGraph, min_dist_m :float = 50, max_dist_m :float = 100, max_tries : int = 10000):
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
        
    def edge_cost(self, e: EdgeKey) -> float:
        """
        Dynamic edge cost: closed edges become impassable (infinite cost).
        """
        u, v, k = e
        data = self.graph.edges[u, v, k]
        if not data.get("open", True):
            return float("inf")

        base = float(data.get("base_cost", 1.0))
        mult = float(data.get("cost_mult", 1.0))
        add = float(data.get("cost_add", 0.0))
        cost = base * mult + add

        # Guard against zero/negative costs
        return max(cost, 1e-9)

    def outgoing_edges(self, u: int) -> List[EdgeKey]:
        """
        Return outgoing directed edges (u -> v) as (u,v,key).
        """
        out = []
        for _, v, k in self.graph.out_edges(u, keys=True):
            out.append((u, v, k))
        return out

    def path_cost(self, path_edges: List[EdgeKey]) -> float:
        return sum(self.edge_cost(e) for e in path_edges)


    def close_edge(self, u: int, v: int, key: int = 0) -> None:
        self.graph.edges[u, v, key]["open"] = False


    def open_edge(self, u: int, v: int, key: int = 0) -> None:
        self.graph.edges[u, v, key]["open"] = True


    def set_edge_multiplier(self, u: int, v: int, key: int, mult: float) -> None:
        self.graph.edges[u, v, key]["cost_mult"] = float(mult)

    def add_edge_penalty(self, u: int, v: int, key: int, penalty: float) -> None:
        self.graph.edges[u, v, key]["cost_add"] = float(penalty)

    def close_edges_by_osmid(self, osmid: int) -> int:
        """
        Convenience: close all edges whose 'osmid' matches (or contains) osmid.
        Returns number of edges affected.
        Note: OSMnx edges can have osmid as int or list.
        """
        n = 0
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            oid = data.get("osmid", None)
            if oid is None:
                continue
            if oid == osmid or (isinstance(oid, (list, tuple, set)) and osmid in oid):
                data["open"] = False
                n += 1
        return n
    
    def nearest_node(self, lat: float, lon: float) -> int:
        return int(ox.distance.nearest_nodes(self.graph, X=lon, Y=lat))
    
    def edges_to_nodes(self, source: int, path_edges: List[EdgeKey]) -> List[int]:
        """
        Convert edge-list to node-list for plotting.
        """
        nodes = [source]
        for _, v, _ in path_edges:
            nodes.append(v)
        return nodes
