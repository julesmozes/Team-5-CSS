import numpy as np
import matplotlib.pyplot as plt
import minimal.osmap
import minimal.ants
import evolutionary_cycle
from scipy.stats import mannwhitneyu
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

np.random.seed(42)


def load_evolution_backup(path="evolution_backup.npz"):
    """
    Load a saved EvolveColonies backup (.npz).

    Returns a dictionary with:
      - history
      - fitness_history
      - best_individual
      - best_paths
      - final_population
    """
    data = np.load(path, allow_pickle=True)

    return {
        "history": data["history"],
        "fitness_history": data["fitness_history"],
        "best_individual": data["best_individual"],
        "best_paths": list(data["best_paths"]),
        "final_population": data["final_population"],
    }


def initialize_population(number_ants, pop_size):
        """Create an initial population of colonies.
        Gene 0 = alpha pheromone importance
        Gene 1 = beta  heuristic importance
        """
        alpha_low, alpha_high = -2, 5.0
        beta_low, beta_high = 1.0, 5.0

        population = np.empty((pop_size, number_ants, 2), dtype=float)

        population[..., 0] = np.random.uniform(
            alpha_low, alpha_high, size=(pop_size, number_ants)
        )

        population[..., 1] = np.random.uniform(
            beta_low, beta_high, size=(pop_size, number_ants)
        )

        return population

def _evaluate_single_colony(colony_idx, colony, local_map, min_dist, num_evals, num_iterations, num_ants):
    np.random.seed()

    target_dist = min_dist + 50.0
    its_to_threshold = []

    for _ in range(num_evals):
        local_map.resetPheromones()
        reached_it = num_iterations  # default = failure

        for it_idx in range(num_iterations):
            solutions = []

            for ant_idx in range(num_ants):
                path, length, steps = ants.build_path_numba(
                    local_map.getNumbaData(),
                    colony[ant_idx, 0],
                    colony[ant_idx, 1],
                )
                if steps:
                    solutions.append((path, length, steps))

            if not solutions:
                continue

            ants.update_pheromones_numba(
                local_map.pheromone,
                solutions,
                Q=1,
                evaporation=0.2,
            )

            best_length = min(length for _, length, _ in solutions)
            if best_length <= target_dist:
                reached_it = it_idx
                break

        its_to_threshold.append(reached_it)

    return colony_idx, np.asarray(its_to_threshold, dtype=np.int32)


def evaluate_population_parallel(
    map_obj,
    population,
    num_evals,
    num_iterations,
    num_ants=100,
    max_workers=None,
    desc="Evaluating population",
):
    pop_size = len(population)
    results_per_colony = [None] * pop_size

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_single_colony,
                idx,
                population[idx],
                map_obj.copy(),
                min_dist,
                num_evals,
                num_iterations,
                num_ants,
            )
            for idx in range(pop_size)
        ]

        for fut in tqdm(
            as_completed(futures),
            total=pop_size,
            desc=desc,
            ncols=100,
        ):
            colony_idx, its = fut.result()
            results_per_colony[colony_idx] = its

    return np.concatenate(results_per_colony, axis=0)

if __name__ == "__main__":
    np.random.seed(42)

    backup = load_evolution_backup("./new_version_colony_evolution/evolution_backup_final.npz")
    map = osmap.Map()
    map.build("Amsterdam, Netherlands", add_travel_time=False)
    min_dist = map.set_src_dest_from_nodes(460441703, 46276928)
    print(min_dist)

    REPEATS_PER_EVALUATION = 20
    NUM_ITERATIONS = 1000
    final_population = backup["final_population"]

    initial_population = initialize_population(number_ants=100, pop_size=40)

    first_results = evaluate_population_parallel(
        map,
        initial_population,
        REPEATS_PER_EVALUATION,
        NUM_ITERATIONS,
        num_ants=100,
        max_workers=None,
        desc="First generation",
    )

    last_results = evaluate_population_parallel(
        map,
        final_population,
        REPEATS_PER_EVALUATION,
        NUM_ITERATIONS,
        num_ants=100,
        max_workers=None,
        desc="Last generation",
    )

    u_stat, p_value = mannwhitneyu(
        first_results,
        last_results,
        alternative="greater"  
        # H₁: first generation takes MORE iterations than last
    )

    print(f"Mann–Whitney U: {u_stat:.2f}")
    print(f"p-value: {p_value:.4e}")

    def cliffs_delta(x, y):
        n_x = len(x)
        n_y = len(y)
        greater = sum(xi > yi for xi in x for yi in y)
        less = sum(xi < yi for xi in x for yi in y)
        return (greater - less) / (n_x * n_y)

    delta = cliffs_delta(first_results, last_results)
    print(f"Cliff’s delta: {delta:.3f}")

    plt.boxplot(
        [first_results, last_results],
        labels=["First generation", "Last generation"],
        showfliers=False,
    )
    plt.ylabel("Iterations to reach threshold")
    plt.title("Evolutionary improvement test")
    plt.grid(alpha=0.3)
    plt.show()

#Mann–Whitney U: 363006.50
# p-value: 8.4928e-07
# Cliff’s delta: 0.134