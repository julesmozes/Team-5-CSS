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
    # Extract quality values (index 0)
    quality = fitness_history[:, :, 0]
    mean_fitness = quality.mean(axis=1)
    std_fitness = quality.std(axis=1)
    gens = np.arange(len(mean_fitness))

    plt.figure()
    plt.plot(gens, mean_fitness, linewidth=2, label="Mean fitness (quality)")
    plt.fill_between(
        gens,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.3,
        label="Std deviation"
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness (Quality)")
    plt.title("Average population fitness with spread")
    plt.legend()
    plt.show()


def plot_effort(fitness_history):
    # Extract effort values (index 1)
    effort = fitness_history[:, :, 1]
    mean_effort = effort.mean(axis=1)
    std_effort = effort.std(axis=1)
    gens = np.arange(len(mean_effort))

    plt.figure()
    plt.plot(gens, mean_effort, linewidth=2, label="Mean effort (iterations)")
    plt.fill_between(
        gens,
        mean_effort - std_effort,
        mean_effort + std_effort,
        alpha=0.3,
        label="Std deviation"
    )

    plt.xlabel("Generation")
    plt.ylabel("Effort (Iterations to convergence)")
    plt.title("Average population effort with spread")
    plt.legend()
    plt.show()

def plot_quality_times_effort(fitness_history):
    # Extract effort values (index 1)
    effort = fitness_history[:, :, 1]
    quality = fitness_history[:, :, 0]
    quality_effort = effort*quality
    mean = quality_effort.mean(axis=1)
    std = quality_effort.std(axis=1)
    gens = np.arange(len(mean))

    plt.figure()
    plt.plot(gens, mean, linewidth=2, label="Mean effort (iterations)")
    plt.fill_between(
        gens,
        mean - std,
        mean + std,
        alpha=0.3,
        label="Std deviation"
    )

    plt.xlabel("Generation")
    plt.ylabel("Effort*Quality")
    plt.title("Average population effort with spread")
    plt.legend()
    plt.show()


def main():
    colony_population_evolution = EvolveColonies(number_colonies=20,
                                                ants_per_colony=150,
                                                generations=10, 
                                                iterations=400, 
                                                mutation_rate=0.2, 
                                                place="Amsterdam, Netherlands", 
                                                add_travel_time=False,
                                                min_distance=6000,
                                                max_distance=7000,
                                                single_path=True)
    history, fitness_history = colony_population_evolution.run(plot_dir="plots", plot_each_generation=True, show_plots=False)

    # Collapse ants + colonies
    pheromone = history[..., 0]  # shape: (gen, col, ant)
    heuristic = history[..., 1]
    plot_param(pheromone, "Pheromone importance")
    plot_param(heuristic, "Heuristic importance")
    plot_fitness(fitness_history)
    plot_effort(fitness_history)
    plot_quality_times_effort(fitness_history)


if __name__ == "__main__":
    main()