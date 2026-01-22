import numpy as np
import random
np.random.seed(42)

# evolution happens on a the level of a whole colony.
# The genetic code of the colony consists of an array that contains the parameters that are used in the next state formula for each ant.
# that is:
# for each ant: [[Pheromone importance, Heuristic importance]]


def initialize_population(number_ants, pop_size):
    """Create an initial population of colonies, the DNA is for the ants in the colonies"""
    return np.random.rand(pop_size, number_ants, 2)

def fitness(population):
    """Evaluate how good a population is."""
    return population.mean(axis=(1, 2)) # placeholder

def select(population, fitnesses, num_offspring):
    """Select parents based on fitness."""
    probs = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), size=num_offspring, p=probs)
    return population[idx]

def crossover(parents):
    """Combine two parent conlonies to produce a child colony"""
    num_offspring, _, n_ants, _ = parents.shape
    cuts = np.random.randint(1, n_ants, size=num_offspring)

    children = parents[:, 0].copy()
    idx = np.arange(n_ants)[None, :] >= cuts[:, None]
    children[idx] = parents[:, 1][idx]
    return children

def mutate(population, mutation_rate=0.1):
    """Randomly perturb colonies by some rate"""
    mask = np.random.rand(len(population)) < mutation_rate
    noise = np.random.uniform(-0.1, 0.1, size=population.shape)
    population[mask] += noise[mask]
    return population

def evolutionary_algorithm(
    pop_size=50,
    generations=100,
    mutation_rate=0.1,
    number_ants=100,
):
    population = initialize_population(number_ants, pop_size)
    best_individual = np.zeros((generations, number_ants, 2))

    for gen in range(generations):
        fitnesses = fitness(population)
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_individual[gen] = population[best_idx]
        
        parents = select(population, fitnesses, pop_size * 2)
        parents = parents.reshape(pop_size, 2, number_ants, 2)

        population = crossover(parents)
        population = mutate(population, mutation_rate)

        if gen % 100 == 0:
            print(f"Gen {gen:05d} | Best fitness: {best_fitness:.6f}")

    return population

evolutionary_algorithm(1000, 10000, 0.1)