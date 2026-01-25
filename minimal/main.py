import numpy as np
import osmap 
import ants

map = osmap.Map()

map.build("Oost, Amsterdam, Netherlands")
min_dist = map.makeRandomTest(300, 500, plot=False)

print("minimum distance Dijkstra: ", min_dist)

N_ANTS = 200
ITER = 1000
alpha = 2
beta = 1.5
Q = 1
evaporation = 0.2

# # kleine genepool test
# alphas = alpha + 1 * np.random.random(N_ANTS)
# betas = beta + 1 * np.random.random(N_ANTS)

alphas = alpha * np.ones(N_ANTS)
betas = beta * np.ones(N_ANTS)

for it in range(ITER):
    solutions = []

    for i in range(N_ANTS):
        path, length = ants.build_path(map, alphas[i], betas[i])
        if path:
            solutions.append((path, length))

    if not solutions:
        print("no solutions")
        continue

    ants.update_pheromones(map, solutions, Q, evaporation)

    best = min(length for _, length in solutions)
    print(f"iter {it}, best length {best:.3f}")