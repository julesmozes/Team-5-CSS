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

for it in range(ITER):
    solutions = []

    for i in range(N_ANTS):
        path, length, steps = ants.build_path_numba(map.getNumbaData(), alpha, beta, max_steps=1000)
        if steps:
            solutions.append((path, length, steps))

    if not solutions:
        print(f"iter {it}, no solution")
        continue

    ants.update_pheromones_numba(map.pheromone, solutions, Q, evaporation)

    best = min(length for _, length, _ in solutions)
    print(f"iter {it}, best length {best:.3f}")