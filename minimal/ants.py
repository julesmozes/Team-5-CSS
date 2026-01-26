import numpy as np
from numba import njit

def build_path(map, alpha, beta, max_steps=1000):
    G = map.G
    start = map.src
    end = map.dest

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
            
            tau = edge["pheromone"]
            eta = edge["heuristic"]

            p = (tau ** alpha) * (eta ** beta)
            probs.append(p)

        probs = np.array(probs)
        s = probs.sum()
        # # guard against NaN / inf / zero-sum
        # if not np.isfinite(s) or s <= 0.0:
        #     probs = np.ones(len(neighbors)) / len(neighbors)
        # else:
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
    
    # Track which edges were used
    used_edges = set()
    for path, length in paths:
        for u, v in path:
            used_edges.add((u, v))
    
    # Only evaporate used edges
    for u, v in used_edges:
        G[u][v]["pheromone"] = max((1 - evaporation) * G[u][v]["pheromone"], 1e-12)
    
    # Deposit on used edges
    for path, length in paths:
        deposit = Q / length
        for u, v in path:
            G[u][v]["pheromone"] += deposit

@njit
def build_path_numba(
    mapData,
    alpha,
    beta,
    max_steps=10000
):  
    
    (max_deg,
    neighbors,
    degree,
    heuristic,
    pheromone,
    cost,
    start,
    end) = mapData

    path = np.empty((max_steps, 2), dtype=np.int32)

    current = start
    last = -1
    length = 0.0
    steps = 0

    for _ in range(max_steps):
        if current == end:
            break

        # compute probabilities
        total = 0.0
        probs = np.zeros(max_deg, dtype=np.float64)

        for k in range(degree[current]):
            v = neighbors[current, k]
            if v == last:
                continue

            w = (pheromone[current, k] ** alpha) * \
                (heuristic[current, k] ** beta)

            probs[k] = w
            total += w

        if total == 0.0:
            return path, 0, 0  # dead end

        # roulette wheel selection
        r = np.random.rand() * total
        cum = 0.0
        chosen_k = -1

        for k in range(degree[current]):
            cum += probs[k]
            if r <= cum:
                chosen_k = k
                break

        next_node = neighbors[current, chosen_k]

        path[steps, :] = [current, chosen_k] 
        length += cost[current, chosen_k]

        last = current
        current = next_node
        steps += 1

    if steps <= max_steps:
        return path, length, steps
    else:
        return path, 0, 0

@njit
def update_pheromones_numba(pheromones, paths, Q, evaporation):
    pheromones *= (1 - evaporation)

    for path, length, steps in paths:
        deposit = Q / length
        for i in range(steps):
            pheromones[path[i, 0], path[i, 1]] += deposit

    return
