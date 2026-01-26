import numpy as np
import random

import minimal.ants as ants
import minimal.osmap as osmap
from concurrent.futures import ProcessPoolExecutor

np.random.seed(42)


# evolution happens on a the level of a whole colony.
# The genetic code of the colony consists of an array that contains the parameters that are used in the next state formula for each ant.
# that is:
# for each ant: [[Pheromone importance, Heuristic importance]]
class EvolveColonies:
    def __init__(self, number_colonies: int, generations: int, mutation_rate: float, ants_per_colony: int, place: str = "Oost, Amsterdam, Netherlands", network_type: str = "drive", add_travel_time: bool = True, evaporation_rate:float=0.2, Q: float = 1, iterations=200):
        self.pop_size = number_colonies
        self.generations = generations
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.num_ants = ants_per_colony
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.map = osmap.Map()
        self.map.build("Oost, Amsterdam, Netherlands")
        
        # HISTORY, used for plotting/restoring
        self.history = np.zeros(
            (generations, number_colonies, ants_per_colony, 2),
            dtype=float
        )
        # Fitness history, used for plotting/restoring
        self.fitness_history = np.zeros(
            (generations, number_colonies),
            dtype=float
        )

        self.population = self.initialize_population(ants_per_colony, number_colonies)
        self.best_individual = np.zeros((generations, ants_per_colony, 2))


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
    def _dominates(q_a, e_a, q_b, e_b):
        """True if A Pareto-dominates B (minimize both objectives)."""
        return (q_a <= q_b and e_a <= e_b) and (q_a < q_b or e_a < e_b)
    
    def select_pareto(self, population, quality, effort, num_offspring, k=3):
        selected = []
        n = len(population)

        for _ in range(num_offspring):
            competitors = np.random.choice(n, size=k, replace=False)

            # Find non-dominated competitors
            nondominated = []
            for i in competitors:
                dominated = False
                for j in competitors:
                    if j == i:
                        continue
                    if self._dominates(quality[j], effort[j], quality[i], effort[i]):
                        dominated = True
                        break
                if not dominated:
                    nondominated.append(i)

            # Tie-break: pick best quality, then best effort
            nd = np.array(nondominated, dtype=int)
            # lexsort sorts by last key first -> (effort, quality) gives quality primary if placed last
            order = np.lexsort((effort[nd], quality[nd]))
            winner = nd[order[0]]

            selected.append(population[winner])

        return np.array(selected)

    def _evaluate_colony(self, colony_idx, colony, local_map):
        np.random.seed()  # uses OS entropy
        best_overall = np.inf
        best_it = self.iterations  # default if never improves

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

            ants.update_pheromones_numba(local_map.pheromone, solutions, self.Q, self.evaporation_rate)

            best = min(length for _, length, _ in solutions)
            if best < best_overall:
                best_overall = best
                best_it = it_idx

        # guard: if never found a solution
        if not np.isfinite(best_overall):
            best_overall = 1e18
            best_it = self.iterations

        return colony_idx, best_overall, best_it

    
    def fitness(self, population):
        min_dist = self.map.makeRandomTest(min_dist=500, max_dist=800, plot=False)
        print("minimum distance Dijkstra: ", min_dist)

        quality = np.empty(self.pop_size, dtype=float)
        effort = np.empty(self.pop_size, dtype=float)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_colony, idx, population[idx], self.map.copy())
                for idx in range(self.pop_size)
            ]

            for future in futures:
                colony_idx, best_overall, best_it = future.result()
                quality[colony_idx] = best_overall / min_dist # lower is better
                effort[colony_idx] = best_it  # lower is better

        return quality, effort


    def select(self, population, fitnesses, num_offspring, k=3):
        """Select parents based on fitness."""
        selected = []
        n = len(population)

        for _ in range(num_offspring):
            competitors = np.random.choice(n, size=k, replace=False)
            best = competitors[np.argmin(fitnesses[competitors])]
            selected.append(population[best])

        return np.array(selected)

    def crossover(self, parents):
        """Combine two parent conlonies to produce a child colony"""
        num_offspring, _, n_ants, _ = parents.shape
        cuts = np.random.randint(1, n_ants, size=num_offspring)

        children = parents[:, 0].copy()
        idx = np.arange(n_ants)[None, :] >= cuts[:, None]
        children[idx] = parents[:, 1][idx]
        return children

    def mutate(self, population, mutation_rate=0.1):
        """Randomly perturb colonies by some rate"""
        # possible improvement: log scale mutation
        mask = np.random.rand(*population.shape) < mutation_rate
        noise = np.random.uniform(-0.1, 0.1, size=population.shape)
        population[mask] += noise[mask]
        population[..., 0] = np.clip(population[..., 0], -5.0, 5.0)   # alpha
        population[..., 1] = np.clip(population[..., 1], 0.0, 10.0)   # beta
        return population
    
    def save(self, path="evolution_backup.npz"):
        np.savez_compressed(
            path,
            history=self.history,
            fitness_history=self.fitness_history,
            best_individual=self.best_individual,
            final_population=self.population
        )
    
    def run(self):
        for gen in range(self.generations):
            quality, effort = self.fitness(self.population)
            print("quality:", quality)
            print("effort:", effort)

            # If you still want a single number for logging:
            # (quality is the primary thing)
            best_idx = np.argmin(quality)
            best_quality = quality[best_idx]
            best_effort = effort[best_idx]

            # ---- STORE HISTORY ----
            self.history[gen] = self.population
            # store just quality as your "fitness_history" (or expand it)
            self.fitness_history[gen] = quality
            self.save()

            self.best_individual[gen] = self.population[best_idx]

            elite = self.population[best_idx:best_idx+1]

            parents = self.select_pareto(self.population, quality, effort, self.pop_size * 2, k=3)
            parents = parents.reshape(self.pop_size, 2, self.num_ants, 2)

            self.population = self.crossover(parents)
            self.population = self.mutate(self.population, self.mutation_rate)
            self.population[0] = elite[0]

            print(f"Gen {gen:05d} | Best quality: {best_quality:.6f} | Effort(it): {best_effort}")

        return self.history, self.fitness_history


