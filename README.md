# Team 5: Evolutionary Complex System Simulation
## Description and motivation
In this project, we take a deep dive into Ant Colony Optimization. It is a complex system in which simple local rules followed by individual agents lead to the emergence of a global shortest path. Ant Colony Optimization is inspired by the behavior of real ant colonies, where ants explore their environment and communicate indirectly through pheromone trails, gradually reinforcing efficient paths while less optimal ones fade away.

We apply this approach to a real-world map using OpenStreetMap data. This topic is interesting because, much like in real life, biological complex systems (such as ant colonies) did not become effective by pure chance. Their behavior was refined through years of evolution, adaptation, and competition. This project aims to mimic that real-world process by using an evolutionary algorithm. Multiple colonies compete with each other, the best-performing colonies persist over generations, and mutations introduce variation that allows further improvement.

Next, we extend the project to a dynamic map environment. We demonstrate that an adaptive, evolutionary colony can outperform a colony with static parameters. As in nature, a system that can adapt and evolve is better suited to changing circumstances and environments. By allowing a colony to adjust over time, the colony is able to find shorter paths on the map, showing a clear benefit of this approach, applicable to a real-world problem.

- see `/colony_evolution/README.md` for research question 1 
- see `/adaptation_evolution/README.md` for research question 2 
- see `/slides/` for presentation
- see `/animations/` for code used to generate animations


## Research Questions
1. Is it possible to optimize the local rules of individual ants in ant colony optimisation to reduce the number of iterations needed to find an optimal solution on a static real world map by using an evolutionary algorithm? How does the best ant colony compare to an ant colony optimised using grid search? And how does the final ant population compare to an ant colony population that was initialised random?
2. Can an evolutionary ant colony with two genomes outperform a regular ant colony in terms of adaptation to a dynamic network?

Hypotheses:
1. a. It is possible to optimize the local rules of individual ants in ant colony optimisation to reduce the number of iterations needed to find an optimal solution using an evolutionary algorithm.

To prove this we will compare the performance of the final ant colony population to an ant colony population that was initialised random.

1. b. The best Ant Colony from our evolutionary alogrithm will outperform an Ant Colony that shares one alpha and beta value, optimized using grid search.

2. We expect the increased exploratory nature of the two-genome colony will provide an improvement in terms of adaptation compared to the regular single-genome colony.

## Limitations
Due to the short time of the project we don't have much time to optimize the large number of parameters involved in the algorithms.
Also we lack a lot of computing power. But even with these limitation we were able to demonstrate a significant improvement, so maybe we could demonstrate an even better improvement when these limitations can be handled.
