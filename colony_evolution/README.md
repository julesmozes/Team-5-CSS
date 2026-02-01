# This folder answers RQ1:
Is it possible to optimize the local rules of individual ants in ant colony optimisation to reduce the number of iterations needed to find an  optimal solution on a static real world map by using an evolutionary algorithm? How does the best ant colony compare to an ant colony optimised using grid search? And how does the final ant population compare to an ant colony population that was initialised random?

## overview
Entry point -> run `rq1.py`
make more plots and compare with random population -> run `analyze_data.py`

## How it works
As explained in the main `readme` this part tries to mimick real-world biological evolution in ant colonies. We argue that this is a big driver of why complex systems can become as good as they are, in real life.

The approach combines Ant Colony Optimization with an evolutionary algorithm to automatically discover effective colony behavior for path finding on a real-world map.

Each individual in the population represents an entire ant colony rather than a single solution. A colony is defined by the parameters used by its ants, specifically how strongly each ant values pheromone information versus heuristic information when choosing where to move. These parameters are different for each ant, allowing diverse local rules within the same colony.

To evaluate a colony, ants repeatedly attempt to find a path between two locations on a real map. As in classical Ant Colony Optimization, ants construct paths step by step based on pheromone levels and heuristics, and successful paths reinforce pheromone trails. Over many iterations, a shortest path emerges from these simple local rules. The performance of a colony is measured by how quickly it finds a path that is close to the optimal shortest path. Colonies that require fewer iterations to reach a good solution are considered fitter.

Evolution happens at the colony level. In each generation, multiple colonies are evaluated independently. The best-performing colonies are more likely to be selected as parents for the next generation. New colonies are created by copying these parents and applying mutations to their parameters. The mutation process is carefully controlled so that parameters remain within valid bounds while allowing both small refinements and occasional larger changes. Over time, mutation strength is reduced, encouraging exploration early on and fine-tuning later.

To maintain diversity and avoid premature convergence, the algorithm preserves the best colony unchanged each generation and introduces a completely new random colony as an immigrant.

## Analysis
The analysis evaluates whether the evolutionary process actually improves colony performance over time.

First, the final state of the evolutionary run is loaded from disk. This includes the fully evolved population of colonies produced after all generations. For comparison, a fresh initial population is also created using the same random initialization procedure that was used at the start of evolution. This allows a fair baseline comparison between unevolved and evolved colonies.

Both the initial and final populations are evaluated on the same fixed routing problem on a real map. For each colony, multiple independent trials are run. In each trial, ants repeatedly search for a path until they find one that is close to the optimal shortest path. The number of iterations required to reach this threshold is recorded. Lower values indicate better performance, since the colony converges more quickly to a good solution.

All evaluations are performed in parallel to reduce variance from runtime effects and to allow many repetitions. The result is a distribution of convergence times for the first generation and for the final evolved generation.

To compare these two distributions, a non-parametric statistical test is used. The Mann–Whitney U test checks whether colonies from the first generation require more iterations on average than colonies from the last generation. This test does not assume normality and is well suited for comparing algorithmic performance distributions. A very small p-value indicates that the observed improvement is statistically significant.

In addition to statistical significance, the effect size is measured using Cliff’s delta. This quantifies how often a randomly chosen initial colony performs worse than a randomly chosen evolved colony. This provides an interpretable measure of practical improvement rather than just significance.

The results are visualized using a box plot, showing the shift in convergence speed from the first generation to the last. This analysis demonstrates that evolution leads to faster convergence and more effective emergence of short paths.

Finally, we investigate whether the best performing ant colony can outperform an ant colony that uses one value for heurstic and pheromone importance, optimized using grid-search. We unfortunately couldn't prove this at this time.