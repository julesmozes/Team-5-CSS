import numpy as np
import random

import ants
import osmap
from concurrent.futures import ProcessPoolExecutor
import osmnx as ox
import matplotlib.pyplot as plt

np.random.seed(42)

# fitness will be tries to find a path close to optimal (within 50m)

class EvolveColonies:
    def __init__(self, number_colonies: int, generations: int, mutation_rate: float, ants_per_colony: int, place: str = "Oost, Amsterdam, Netherlands", add_travel_time: bool = False, evaporation_rate: float=0.2, Q: float = 1, iterations=200, min_distance=800, max_distance=1000, single_path=True):
        self.pop_size = number_colonies
        self.generations = generations
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.num_ants = ants_per_colony
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.map = osmap.Map()
        self.map.build(place, add_travel_time=add_travel_time)
        self.min_distance_random = min_distance
        self.max_distance_random = max_distance
        self.repeats_per_evaluation = 5

        self.alpha_min, self.alpha_max = -5.0, 5.0
        self.beta_min, self.beta_max = 0.0, 10.0

        self.sigma0 = 0.35     # starting noise
        self.sigma_decay = 3.0 # higher = faster decay
        self._eps = 1e-6       # numerical stability for transforms

        self.follow_single_path = single_path
        if self.follow_single_path:
            self.min_dist = self.map.pick_node_pair_interactive(plot=False)  

        # HISTORY, used for plotting/restoring
        self.history = np.zeros(
            (generations, number_colonies, ants_per_colony, 2),
            dtype=float
        )
        # Fitness history, used for plotting/restoring
        self.fitness_history = np.zeros(
            (generations, number_colonies, 1),
            dtype=float
        )
        # Per generation, the best path for each colony (list-of-lists of node ids)
        self.best_paths = [None] * generations

        self.population = self.initialize_population(ants_per_colony, number_colonies)
        self.best_individual = np.zeros((generations, ants_per_colony, 2))
    
    def _sigma(self, gen: int) -> float:
        # annealed sigma
        return float(self.sigma0 * np.exp(-self.sigma_decay * gen / max(1, self.generations - 1)))

    def mutate(self, population, gen: int, mutation_rate=None):
        """
        Logit-normal (bounded) Gaussian mutation:
        - Map gene to (0,1)
        - Add Gaussian noise in logit space (log-scale-like)
        - Map back to original bounds

        Works for negative alpha because we mutate in a bounded transformed space.
        """
        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        sigma = self._sigma(gen)

        # mask: which genes to mutate
        mask = (np.random.rand(*population.shape) < mutation_rate)

        # helpers
        def to_unit(x, lo, hi):
            u = (x - lo) / (hi - lo)
            return np.clip(u, self._eps, 1.0 - self._eps)

        def logit(u):
            return np.log(u / (1.0 - u))

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        # --- alpha channel ---
        alpha = population[..., 0]
        u = to_unit(alpha, self.alpha_min, self.alpha_max)
        z = logit(u)

        noise = np.random.normal(0.0, sigma, size=alpha.shape)
        z_mut = z + noise

        u_mut = sigmoid(z_mut)
        alpha_new = self.alpha_min + u_mut * (self.alpha_max - self.alpha_min)

        # apply only where masked for alpha
        alpha_mask = mask[..., 0]
        alpha[alpha_mask] = alpha_new[alpha_mask]
        population[..., 0] = alpha

        # --- beta channel ---
        beta = population[..., 1]
        u = to_unit(beta, self.beta_min, self.beta_max)
        z = logit(u)

        noise = np.random.normal(0.0, sigma, size=beta.shape)
        z_mut = z + noise

        u_mut = sigmoid(z_mut)
        beta_new = self.beta_min + u_mut * (self.beta_max - self.beta_min)

        beta_mask = mask[..., 1]
        beta[beta_mask] = beta_new[beta_mask]
        population[..., 1] = beta

        # final clamp (just in case)
        population[..., 0] = np.clip(population[..., 0], self.alpha_min, self.alpha_max)
        population[..., 1] = np.clip(population[..., 1], self.beta_min, self.beta_max)

        return population


    def initialize_population(self, number_ants, pop_size):
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

    @staticmethod
    def _path_indices_to_nodes(path, steps, neighbors, nodes):
        """Convert numba path (node, neighbor_idx) to a list of node ids."""
        if steps == 0:
            return None

        route = [int(nodes[path[0, 0]])]
        for i in range(steps):
            u = path[i, 0]
            k = path[i, 1]
            v = neighbors[u, k]
            route.append(int(nodes[v]))

        return route

    def _evaluate_colony(self, colony_idx, colony, local_map, min_dist):
        np.random.seed()  # uses OS entropy

        its_to_threshold = []
        best_path_nodes = None

        target_dist = min_dist + 50.0

        for evaluation_idx in range(self.repeats_per_evaluation):
            local_map.resetPheromones()

            reached_it = self.iterations  # default = failure

            for it_idx in range(self.iterations):
                solutions = []

                for ant_idx in range(self.num_ants):
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
                    self.Q,
                    self.evaporation_rate,
                )

                best_idx = np.argmin([length for _, length, _ in solutions])
                best_length = solutions[best_idx][1]

                # stop as soon as we're close enough
                if best_length <= target_dist:
                    reached_it = it_idx

                    # keep the path that achieved success
                    best_path_nodes = self._path_indices_to_nodes(
                        solutions[best_idx][0],
                        solutions[best_idx][2],
                        local_map.neighbors,
                        local_map.nodes,
                    )
                    break

            its_to_threshold.append(reached_it)

        avg_effort = float(np.mean(its_to_threshold))
        return colony_idx, avg_effort, best_path_nodes

    
    def fitness(self, population):
        if self.follow_single_path:
            min_dist = self.min_dist
        else:
            min_dist = self.map.makeRandomTest(
                min_dist=self.min_distance_random,
                max_dist=self.max_distance_random,
                plot=False,
            )

        print("minimum distance Dijkstra: ", min_dist)

        effort = np.empty(self.pop_size, dtype=float)
        paths = [None] * self.pop_size

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._evaluate_colony,
                    idx,
                    population[idx],
                    self.map.copy(),
                    min_dist,
                )
                for idx in range(self.pop_size)
            ]

            for future in futures:
                colony_idx, avg_effort, best_path_nodes = future.result()
                effort[colony_idx] = avg_effort
                paths[colony_idx] = best_path_nodes

        # fitness = iterations needed (lower is better)
        return effort, paths



    def select(self, population, fitnesses, num_offspring, k=4):
        """Select parents based on fitness."""
        selected = []
        n = len(population)

        for _ in range(num_offspring):
            competitors = np.random.choice(n, size=k, replace=False)
            best = competitors[np.argmin(fitnesses[competitors])]
            selected.append(population[best])

        return np.array(selected)

    def save(self, path="evolution_backup.npz"):
        np.savez_compressed(
            path,
            history=self.history,
            fitness_history=self.fitness_history,
            best_individual=self.best_individual,
            best_paths=np.array(self.best_paths, dtype=object),
            final_population=self.population
        )

    def plot_generation_paths(self, gen_idx, route_color="tab:red", save_path=None, show=True):
        """Plot best path of each colony for a given generation."""

        if gen_idx < 0 or gen_idx >= self.generations:
            raise ValueError("gen_idx out of range")

        gen_paths = self.best_paths[gen_idx]
        if gen_paths is None:
            raise ValueError("No paths recorded for this generation")

        routes = [p for p in gen_paths if p]
        if not routes:
            print("No recorded routes to plot.")
            return

        fig, ax = ox.plot_graph_routes(
            self.map.Gm,
            routes,
            route_colors=route_color,
            route_linewidth=2,
            node_size=0,
            bgcolor="white",
            show=False,
            close=False,
        )
        ax.set_title(f"Generation {gen_idx} best-per-colony routes")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    
    def run(self, plot_each_generation=False, plot_dir=None, route_color="tab:red", show_plots=True):
        if plot_each_generation and plot_dir:
            import os
            os.makedirs(plot_dir, exist_ok=True)

        for gen in range(self.generations):
            effort, gen_paths = self.fitness(self.population)
            print("effort:", effort)
            self.best_paths[gen] = gen_paths

            if plot_each_generation:
                save_path = None
                if plot_dir:
                    save_path = f"{plot_dir}/gen_{gen:04d}.png"
                try:
                    self.plot_generation_paths(
                        gen,
                        route_color=route_color,
                        save_path=save_path,
                        show=show_plots,
                    )
                except Exception as exc:
                    print(f"Plotting failed for generation {gen}: {exc}")

            best_idx = np.argmin(effort)
            best_effort = effort[best_idx]

            self.history[gen] = self.population
            self.fitness_history[gen, :, 0] = effort
            self.save()

            self.best_individual[gen] = self.population[best_idx]

            elite = self.population[best_idx:best_idx+1].copy()

            # One random immigrant each generation (replaces a slot in next population)
            immigrant = self.initialize_population(self.num_ants, 1)

            # Create offspring by selecting parents and mutating copies
            # We need pop_size - 2 offspring: 1 elite + (pop-2 offspring) + 1 immigrant
            num_offspring = self.pop_size - 2
            parents = self.select(self.population, effort, num_offspring, k=5)

            offspring = parents.copy()
            offspring = self.mutate(offspring, gen=gen)  # uses self.mutation_rate by default

            # Next generation population
            self.population = np.concatenate([elite, offspring, immigrant], axis=0)

            # Ensure elite is exactly preserved in case of accidental mutation
            self.population[0] = elite[0]


            print(f"Gen {gen:05d} | Best Effort(it): {best_effort}")

        return self.history, self.fitness_history

