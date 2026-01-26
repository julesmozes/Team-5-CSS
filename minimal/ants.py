import numpy as np

def build_path(map, alpha, beta, max_steps=10000):
    G = map.G
    start = map.src
    end = map.dest

    EPS = 1e-12  # numerical safety floor

    path = []
    current = start
    lastNode = start

    steps = 0
    length = 0
    while current != end and steps <= max_steps:
        neighbors = [
            v for v in G.neighbors(current)
            if v != lastNode
        ]
        if not neighbors:
            return None, None  # dead end

        probs = []
        for v in neighbors:
            edge = G[current][v]
            
            tau = max(edge["pheromone"], EPS)   # allows negative alpha
            eta = max(edge["heuristic"], EPS)

            p = (tau ** alpha) * (eta ** beta)
            probs.append(p)

        probs = np.array(probs)
        s = probs.sum()
        # guard against NaN / inf / zero-sum
        if not np.isfinite(s) or s <= 0.0:
            probs = np.ones(len(neighbors)) / len(neighbors)
        else:
            probs /= s

        next_node = np.random.choice(neighbors, p=probs)
        length += G[current][next_node]["cost"]
        path.append((current, next_node))

        lastNode = current
        current = next_node  
        steps += 1

    return path, length

def update_pheromones(map, paths, Q, evaporation):
    G = map.G

    # evaporation
    for u, v in G.edges:
        G[u][v]["pheromone"] *= (1 - evaporation)

    # deposit
    for path, length in paths:
        deposit = Q / length
        for u, v in path:
            G[u][v]["pheromone"] += deposit

    return