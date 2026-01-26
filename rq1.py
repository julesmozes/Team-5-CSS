"""
RQ1, concept: Hoe evolueert de genenpool van de genetische ant colony met optimale waardes voor de normale ant colony?

RQ1: How does the performance and genepool of genetic ant colonies compare to an ant colony with single optimized alpha and beta values?
 - spread of genes plot
 - performance plot, performance over time.

"""
from evolutionary_cycle import EvolveColonies
import matplotlib.pyplot as plt
import numpy as np

def plot_param(param, name):
    mean = param.mean(axis=(1, 2))
    std = param.std(axis=(1, 2))

    plt.figure()
    plt.plot(mean, label=f"Mean {name}")
    plt.fill_between(
        range(len(mean)),
        mean - std,
        mean + std,
        alpha=0.3,
        label="Std dev"
    )
    plt.xlabel("Generation")
    plt.ylabel(name)
    plt.title(f"{name} evolution")
    plt.legend()
    plt.show()

def plot_fitness(fitness_history):
    mean_fitness = fitness_history.mean(axis=1)
    std_fitness = fitness_history.std(axis=1)
    gens = np.arange(len(mean_fitness))

    plt.figure()
    plt.plot(gens, mean_fitness, linewidth=2, label="Mean fitness")
    plt.fill_between(
        gens,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.3,
        label="Std deviation"
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Average population fitness with spread")
    plt.legend()
    plt.show()


def main():
    colony_population_evolution = EvolveColonies(number_colonies=10, ants_per_colony=100, generations=10, iterations=100, mutation_rate=0.1)
    history, fitness_history = colony_population_evolution.run()

    # Collapse ants + colonies
    pheromone = history[..., 0]  # shape: (gen, col, ant)
    heuristic = history[..., 1]
    plot_param(pheromone, "Pheromone importance")
    plot_param(heuristic, "Heuristic importance")
    plot_fitness(fitness_history)


if __name__ == "__main__":
    main()