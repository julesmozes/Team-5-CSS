import minimal.ants as ants
import minimal.osmap as osmap
import numpy as np

map = osmap.Map()

map.build("Oost, Amsterdam, Netherlands")
min_dist = map.makeRandomTest(600, 800, plot=True)

print("minimum distance Dijkstra: ", min_dist)

n_ants = 200
iterations = 1000
alpha = 2
beta = 1.5
Q = 1
evaporation = 0.2

def run_grid_search(alphas, betas, N_ants, iterations, Q, evaporation, map, repeats=3, base_seed=67):
    results = {}
    for alpha in alphas:
        for beta in betas:
            best_lengths = []
            mean_norms = []

            for r in range(repeats):
                np.random.seed(base_seed + r)
                map.resetPheromones()

                best_length = float('inf')
                all_lengths = []

                for it in range(iterations):
                    solutions = []

                    for i in range(N_ants):
                        path, length, steps = ants.build_path_numba(map.getNumbaData(), alpha, beta, max_steps=1000)
                        if steps:
                            solutions.append((path, length, steps))
                            all_lengths.append(length)

                    if not solutions:
                        continue

                    ants.update_pheromones_numba_max(map.pheromone, solutions, Q, evaporation, max_pheromone=0.5)

                    current_best = min(length for _, length, _ in solutions)
                    if current_best < best_length:
                        best_length = current_best

                best_lengths.append(best_length)
                
                mean_len = np.mean(all_lengths) if all_lengths else float('inf')
                mean_norms.append(mean_len / min_dist)

            results[(alpha, beta)] = {
                "best_length": float(np.mean(best_lengths)),
                "mean_norm": float(np.mean(mean_norms)),
            }
            print(
                f"alpha: {alpha}, beta: {beta}, mean_norm: {results[(alpha, beta)]['mean_norm']:.4f}, "
                f"best_length: {results[(alpha, beta)]['best_length']:.3f}"
            )
    return results

alphas = [3.0, 4.0, 5.0]
betas  = [5]
repeats = 2
iterations = 200
grid_search_results = run_grid_search(alphas, betas, n_ants, iterations, Q, evaporation, map)
print("Grid Search Results:")
for params, metrics in grid_search_results.items():
    print(
        f"Parameters (alpha={params[0]}, beta={params[1]}): "
        f"mean_norm={metrics['mean_norm']:.4f}, best_length={metrics['best_length']:.3f}"
    )

best_params = min(grid_search_results, key=lambda p: grid_search_results[p]["mean_norm"])
print(
    f"Best: alpha={best_params[0]}, beta={best_params[1]}, "
    f"mean_norm={grid_search_results[best_params]['mean_norm']:.4f}"
)

    