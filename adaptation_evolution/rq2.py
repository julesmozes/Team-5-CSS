import numpy as np
import ants
import osmap
import matplotlib.pyplot as plt

np.random.seed(90)

class AdaptiveColony:  
    def __init__(
            self, 
            alpha=1.5,
            beta=1,
            explorerRatio=0.01, # set mutation rate to ensure there are always some explorers
            N_ants=100,
            iterations=200,     # amount of iterations before updating the map
            timeSteps=100,      # amount of times the map is updated
            Q=10,               # pheromone deposition for regular ants
            Qexplorer=100,      # pheromone deposition for explorer ants
            evaporation=.5           
        ):
        
        self.N_ants = N_ants
        self.explorerRatio = explorerRatio

        self.Q = Q  
        self.Qexplorer = Qexplorer
        self.evaporation = evaporation

        self.iterations = iterations
        self.timeSteps = timeSteps

        self.alpha = alpha
        self.beta = beta

        self.bestSolutions = np.empty(timeSteps)
        self.baselineSolutions = np.empty(timeSteps)
        self.explorerCounts = np.empty((timeSteps, iterations))

        return
    
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(self.bestSolutions, label="colony with explorers")
        ax1.plot(self.baselineSolutions, linestyle="dashed", color="grey", label="static colony")
        ax1.set_xlabel("time steps")
        ax1.set_ylabel("Best ant solution / optimal solution")
        ax1.legend()

        ax2.plot(np.max(self.explorerCounts, axis=1))
        ax2.set_xlabel("time steps")
        ax2.set_ylabel("Number of explorers")

        plt.show()

        return
    
    def baselineComparison(
            self, 
            map: osmap.Map, 
            reproductionRatio=10    # amount of children per explorer ant, non-explorers get two children
        ):

        map.updateEdges(0)          # set t=0
        map.seedShortestPath()      # seed the best path (given current traffic) with pheromones

        baselineMap = map.copy()

        for g in range(self.timeSteps):
            fitness = np.empty((self.N_ants, self.iterations), dtype=np.float64)
            
            dijkstraDistance = map.getShortestPathLength()

            # baseline colony
            for it in range(self.iterations):
                solutions = []

                for i in range(self.N_ants):
                    path, length, steps = ants.build_path_numba(
                        baselineMap.getNumbaData(),
                        self.alpha,
                        self.beta,
                        max_steps=1000
                    )

                    if steps:
                        solutions.append((path, length, steps))
                        fitness[i, it] = length
                    else:
                        fitness[i, it] = np.nan
                    
                if not solutions:
                    continue
                
                ants.update_pheromones_numba_max(
                    baselineMap.pheromone, 
                    solutions, 
                    self.Q, 
                    self.evaporation, 
                    max_pheromone=.5,
                    min_pheromone=.1
                )

            self.baselineSolutions[g] = np.nanmin(fitness) / dijkstraDistance
            
            alphas = self.alpha * np.ones(self.N_ants)
            alphas *= (np.random.random(self.N_ants) > self.explorerRatio)

            # colony with explorers
            for it in range(self.iterations):
                solutions = []
                Qs = []

                for i in range(self.N_ants):
                    path, length, steps = ants.build_path_numba(
                        map.getNumbaData(),
                        alphas[i],
                        self.beta,
                        max_steps=1000
                    )

                    if steps:
                        solutions.append((path, length, steps))

                        # save Q values for genome-specific deposition
                        if alphas[i]:
                            Qs.append(self.Q)
                        else:
                            Qs.append(self.Qexplorer)

                        fitness[i, it] = length
                    else:
                        fitness[i, it] = np.nan
                    

                if not solutions:
                    continue
                
                ants.update_pheromones_numba_Qs(
                    map.pheromone, 
                    solutions, 
                    Qs, 
                    self.evaporation, 
                    max_pheromone=.5,
                    min_pheromone=.1
                )

                # find top performers based on path length
                ranking = np.argsort(fitness[:, it])
                topPerformers = ranking[:10]
                topPerformerExplorers = alphas[topPerformers] == 0
                topPerformerNonExplorer = alphas[topPerformers] != 0

                # determine amount of children per genome
                N_explorerChildren = reproductionRatio * np.sum(topPerformerExplorers)
                N_nonExplorerChildren = 2 * np.sum(topPerformerNonExplorer)

                # mutate children of non-explorer ants to become explorers
                mutations = np.random.random(N_nonExplorerChildren) > self.explorerRatio

                # replace non-top performers with new offspring
                childrenIndices = np.random.choice(ranking[10:], N_explorerChildren + N_nonExplorerChildren, replace=False)
                alphas[childrenIndices[:N_explorerChildren]] = 0
                alphas[childrenIndices[N_explorerChildren:]] = self.alpha * mutations

                # book-keeping for amount of explorers in colony
                self.explorerCounts[g, it] = np.sum(alphas == 0)
                
            self.bestSolutions[g] = np.nanmin(fitness) / dijkstraDistance     

            print("time-step %d, improvement: %f" % (g, self.bestSolutions[g] - self.baselineSolutions[g]))       
            map.updateEdges(g)
            baselineMap.updateEdges(g)

        return
    

evolutionObject = AdaptiveColony()

map = osmap.Map()
map.build("Oost, Amsterdam, Netherlands")

dijkstraDistance = map.makeRandomTest(500, 800, plot=False)
print("minimum distance Dijkstra: ", dijkstraDistance)

map.buildTraffic(minSpatialWavelength=100, minTemporalWavelength=2, modulationDepth=5)

evolutionObject.baselineComparison(map)

evolutionObject.plot()
                

