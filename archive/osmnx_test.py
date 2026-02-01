from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from classes.antcolony import AntColony, ColonyParams
from classes.maplogic import OSMMap

EdgeKey = Tuple[int, int, int]  # (u, v, key) for MultiDiGraph edges

# ---------------------------
# Helpers for using OSMnx nodes
# ---------------------------

# 1) Build graph
osmmap = OSMMap()

# 2) Pick start/end (example coordinates)
osmmap.set_random_pair()

fig, ax = ox.plot_graph(
    osmmap.graph,
    node_size=0,
    edge_linewidth=0.6,
    bgcolor="white",
    show=False,
    close=False
)

ax.scatter(
    osmmap.graph.nodes[osmmap.src]["x"],
    osmmap.graph.nodes[osmmap.src]["y"],
    c="green",
    s=80,
    zorder=5,
    label="source",
)

ax.scatter(
    osmmap.graph.nodes[osmmap.dst]["x"],
    osmmap.graph.nodes[osmmap.dst]["y"],
    c="red",
    s=80,
    zorder=5,
    label="target",
)

ax.legend()
plt.show()

# 3) Run ACO
colony = AntColony(
    osmmap,
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

best_edges, best_cost = colony.solve(osmmap.src, osmmap.dst)
print("Best cost:", best_cost)


# 4) Visualize result
if best_edges:
    route_nodes = osmmap.edges_to_nodes(osmmap.src, best_edges)
    fig, ax = ox.plot_graph_route(osmmap.graph, route_nodes, route_linewidth=4, node_size=0)