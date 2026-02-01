# this folder answers RQ2:
Can an evolutionary ant colony with two genomes outperform a regular ant colony in terms of adaptation to a dynamic network?
We hypothesise that the two genome colony will better adapt to a changing environment compared to a single genome colony.

## motivation
In a real-life route-finding problem, the optimal solution depends on the traffic conditions. However, traffic information is not readily accesible and often needs to be estimated using models, which can become computationally expensive.
The ant colony maintains an established path through the pheromones and can hopefully adapt to the changing traffic conditions without requiring traffic information about the entire map. From preliminary experimentation we observed
that the regular ant colony does not adapt well to a dynamic environment, especially when new optimal paths are significantly different from the previous ones (e.g. a scenario where a completely different path that used to be 
slightly longer now becomes favourable due to traffic). 
To improve upon the regular ant colony, we introduce a second genome; an 'explorer' ant that has no affinity for pheromone but purely goes by heuristic (geographical distance in this case). The explorer ants provide a benefit, by breaking
away from established paths, but are expensive in terms of the information that they require about the traffic conditions. We try to minimise the amount of explorer ants while retaining their benefits through a natural selection algorithm.

## explanation of the traffic simulation
To simulate changing traffic conditions, we modulate the base edge costs provided by OpenStreetMap using a space-time Fourier series. The Fourier series has 10 components with random coefficients and frequencies, resulting in random
fluctuations in edge costs that are smooth in space and time. The coefficients of the Fourier series are normalised by precomputation of future time steps to facilitate a `modulationDepth` scaling parameter consistent across runs. 
The random spatial/temporal frequencies have a minimum wavelength parameter, which ensures that no excessively high frequency components are introduced in the modulation.

## explanation of the Two-genome colony
The two genome colony has a small concentration of explorer ants that try to find better paths as the traffic conditions change. To regulate the amount of explorers, a natural selection algorithm is used. After every iteration,
the top 10 performers (in terms of shortest path length) are able to reproduce. The non-explorer ants have a reproduction ratio of 2, the explorer ants have a high reproduction ratio of 10. The non-top performer ants are replaced by the new offspring
to ensure a constant colony size.

If the path established by the pheromones
is not optimal, an explorer ant can become a top performer by finding a better path. The high reproduction ratio of the explorer ants makes the entire colony more exploratory in this
case, to hopefully find the optimal path for the current traffic conditions. To ensure there are always some explorer ants (in that they don't go extinct), children of non-explorer ants have a 1% chance of becoming an explorer ant
through mutation.

The natural selection algorithm results in an emergent behaviour: the colony is constantly checking if there is room for improvement. If it finds there is, the colony becomes more exploratory to find the better path and naturally
scales down the number of explorer ants when they are no longer providing benefit.

## setup
We compare the two-genome colony with a regular single-genome colony and evaluate how well they adapt to the changing environment by evaluating how optimal their paths are. The measure for the evaluation of performance is the best path found by the colony divided by the best possible
path as found by Dijkstra's algorithm, indicating how optimal the path is (and by extension, how much room for improvement there is). 

Since we study adaptation rather than regular path finding, we initially seed the optimal path to elminiate warm-up time and to give the two colony types the same starting point. The seeding is done
by having Dijkstra's algorithm find the optimal path given the traffic condtions at time=0 and laying down pheromones on this path.
