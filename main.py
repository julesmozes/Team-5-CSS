import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
    
N_network = 100
p = .0

G = nx.random_geometric_graph(500, 0.2)

for u, v in G.edges:
    G[u][v]["cost"] = np.random.rand() + 0.1
    G[u][v]["pheromone"] = 1.0

def build_path(G, start, end, alpha, beta):
    path = []
    current = start
    visited = {start}

    while current != end:
        neighbors = [
            v for v in G.neighbors(current)
            if v not in visited
        ]
        if not neighbors:
            return None  # dead end

        probs = []
        for v in neighbors:
            edge = G[current][v]
            p = (edge["pheromone"] ** alpha) * ((1 / edge["cost"]) ** beta)
            probs.append(p)

        probs = np.array(probs)
        probs /= probs.sum()

        next_node = np.random.choice(neighbors, p=probs)
        path.append((current, next_node))
        visited.add(next_node)
        current = next_node

    return path

def update_pheromones(G, paths, Q, evaporation):
    # evaporation
    for u, v in G.edges:
        G[u][v]["pheromone"] *= (1 - evaporation)

    # deposit
    for path, length in paths:
        deposit = Q / length
        for u, v in path:
            G[u][v]["pheromone"] += deposit

N_ANTS = 10
ITER = 100
alpha = 1.5
beta = 1.5
Q = 10
evaporation = 0.5

start = np.random.choice(G.nodes)
path = [start]
pos = start

for i in range(50):
    neighbors = [
        v for v in G.neighbors(pos)
        if v not in path
    ]
    if not neighbors:
        break

    path.append(np.random.choice(neighbors))

end = path[-1]

solutions = []
for it in range(ITER):
    solutions = []

    for _ in range(N_ANTS):
        path = build_path(G, start, end, alpha, beta)
        if path:
            length = sum(G[u][v]["cost"] for u, v in path)
            solutions.append((path, length))

    if not solutions:
        continue

    update_pheromones(G, solutions, Q, evaporation)

    best = min(length for _, length in solutions)
    print(f"iter {it}, best length {best:.3f}")

print(min(solutions, key= lambda x : x[1]))




