# R1: Can the genetic ant colony outperform a static ant colony in a traffic distribution problem?
from evolutionary_cycle import EvolveColonies

def main():
    colony_population_evolution = EvolveColonies(number_colonies=10, ants_per_colony=100, generations=30, iterations=200, mutation_rate=0.1)
    colony_population_evolution.run()


if __name__ == "__main__":
    main()