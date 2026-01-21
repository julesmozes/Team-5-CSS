import numpy as np
import random
np.random.seed(42)

# evolution happens on a the level of a whole colony.
# The genetic code of the colony consists of an array that contains the parameters that are used in the next state formula for each ant.
# that is:
# for each ant: [[Pheromone importance, Heuristic importance]]


def initialize_population(number_ants, pop_size):
    """Create an initial population of colonies, the DNA is for the ants in the population"""
    return [np.random.rand(number_ants, 2) for _ in range(0, pop_size)]

def fitness(colony):
    """Evaluate how good an individual is."""
    return colony.mean() # PLACEHOLDER

def select(population, fitnesses, num_parents):
    """Select parents based on fitness."""
    return random.choices(population, weights=fitnesses, k=num_parents)

def crossover(parent1, parent2):
    """Combine two parents to produce a child."""
    assert parent1.shape == parent2.shape
    n_rows = parent1.shape[0]

    # choose cut point (not at extremes)
    cut = np.random.randint(1, n_rows)

    child = np.vstack((
        parent1[:cut],
        parent2[cut:]
    ))

    return child

def mutate(colony, mutation_rate=0.1):
    """Randomly perturb an individual."""
    if np.random.random() < mutation_rate:
        colony += np.random.uniform(-0.1, 0.1)
    return colony

def evolutionary_algorithm(
    pop_size=50,
    generations=100,
    mutation_rate=0.1,
    number_ants=100,
):
    population = initialize_population(number_ants, pop_size)
    best_individual = []

    for gen in range(generations):
        fitnesses = [fitness(colony) for colony in population]
        best_individual.append([population[np.argmax(fitnesses)], np.max(fitnesses)])

        new_population = []
        while len(new_population) < pop_size:
            parents = select(population, fitnesses, num_parents=2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        best_fitness = max(fitnesses)
        print(f"Gen {gen:03d} | Best fitness: {best_fitness:.4f}")

    return population

def tests():
    p1 = np.ones((20, 2))
    p2 = np.zeros((20, 2))

    child = crossover(p1, p2)
    print(child)

evolutionary_algorithm(20, 1000, 0.1)