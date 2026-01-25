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

        self.population = self.initialize_population(ants_per_colony, number_colonies)
        self.best_individual = np.zeros((generations, ants_per_colony, 2))


    def initialize_population(self, number_ants, pop_size):
        """Create an initial population of colonies, the DNA is for the ants in the colonies"""
        # TODO: Add some random range
        low, high = 1, 5
        return np.random.rand(pop_size, number_ants, 2) * (high - low) + low
        # experiment
        # return np.full((pop_size, number_ants, 2), [2, 1.5])

    def _evaluate_colony(self, colony_idx, colony, local_map):
        """
        Evaluate a single colony.
        Returns (colony_idx, fitness)
        """
        best_overall = np.inf

        for it_idx in range(self.iterations):
            solutions = []

            for ant_idx in range(self.num_ants):
                path, length = ants.build_path(
                    local_map,
                    colony[ant_idx, 0],
                    colony[ant_idx, 1],
                )
                if path:
                    solutions.append((path, length))

            if not solutions:
                continue

            ants.update_pheromones(
                local_map, solutions, self.Q, self.evaporation_rate
            )

            best = min(length for _, length in solutions)
            best_overall = min(best_overall, best)

        return colony_idx, best_overall
    
    def fitness(self, population):
        """
        Parallel fitness evaluation.
        Returns fitness array aligned with population order.
        """
        min_dist = self.map.makeRandomTest(100, 200, plot=False)
        print("minimum distance Dijkstra: ", min_dist)
        fitness = np.empty(self.pop_size, dtype=float)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._evaluate_colony,
                    idx,
                    population[idx],
                    self.map.copy()
                )
                for idx in range(self.pop_size)
            ]

            for future in futures:
                colony_idx, value = future.result()
                fitness[colony_idx] = value

        return fitness

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
        mask = np.random.rand(*population.shape) < mutation_rate
        noise = np.random.uniform(-0.1, 0.1, size=population.shape)
        population[mask] += noise[mask]
        return population
    
    def run(self):
        for gen in range(self.generations):
            fitnesses = self.fitness(self.population)
            best_idx = np.argmin(fitnesses)
            best_fitness = fitnesses[best_idx]
            self.best_individual[gen] = self.population[best_idx]

            elite = self.population[best_idx:best_idx+1]
            parents = self.select(self.population, fitnesses, self.pop_size * 2)
            parents = parents.reshape(self.pop_size, 2, self.num_ants, 2)

            self.population = self.crossover(parents)
            self.population = self.mutate(self.population, self.mutation_rate)
            self.population[0] = elite[0]


            print(f"Gen {gen:05d} | Best fitness: {best_fitness:.6f}")

