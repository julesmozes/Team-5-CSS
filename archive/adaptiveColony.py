import numpy as np
import minimal.ants as ants
import minimal.osmap as osmap
import matplotlib.pyplot as plt

np.random.seed(90)

class AdaptiveColony:  
    def __init__(
            self, 
            alpha=1.5,
            beta=1,
            explorerRatio=0.01,
            N_ants=100,
            iterations=200,
            timeSteps=100,
            Q=10,
            Qexplorer=100,
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
    
    def baselineComparison(self, map: osmap.Map, reproductionRatio=5):
        map.updateEdges(0)
        map.seedShortestPath()

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

                ranking = np.argsort(fitness[:, it])
                topPerformers = ranking[:10]
                topPerformerExplorers = alphas[topPerformers] == 0
                topPerformerNonExplorer = alphas[topPerformers] != 0

                N_explorerChildren = reproductionRatio * np.sum(topPerformerExplorers)
                N_nonExplorerChildren = 2 * np.sum(topPerformerNonExplorer)

                mutations = np.random.random(N_nonExplorerChildren) > self.explorerRatio

                childrenIndices = np.random.choice(ranking[10:], N_explorerChildren + N_nonExplorerChildren, replace=False)
                alphas[childrenIndices[:N_explorerChildren]] = 0
                alphas[childrenIndices[N_explorerChildren:]] = self.alpha * mutations

                self.explorerCounts[g, it] = np.sum(alphas == 0)
                
            self.bestSolutions[g] = np.nanmin(fitness) / dijkstraDistance     

            print("time-step %d, improvement: %f" % (g, self.bestSolutions[g] - self.baselineSolutions[g]))       
            map.updateEdges(g)
            baselineMap.updateEdges(g)

        return
    
    def baselineComparison(self, map: osmap.Map, N_explorers, reproductionRatio):
        map.updateEdges(0)
        map.seedShortestPath()

        baselineMap = map.copy()

        uniqueEdgesStatic = np.empty((self.timeSteps, self.iterations))
        uniqueEdgesAdaptive = np.empty((self.timeSteps, self.iterations))

        for g in range(self.timeSteps):
            fitness = np.empty((self.N_ants, self.iterations), dtype=np.float64)
            
            dijkstraDistance = map.getShortestPathLength()
            
            alphas = self.alpha * np.ones(self.N_ants)
            alphas *= (np.random.random(self.N_ants) > self.explorerRatio)

            # colony with static number of explorers

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

                ranking = np.argsort(fitness[:, it])
                topPerformers = ranking[:10]
                topPerformerExplorers = alphas[topPerformers] == 0
                topPerformerNonExplorer = alphas[topPerformers] != 0

                N_explorerChildren = reproductionRatio * np.sum(topPerformerExplorers)
                N_nonExplorerChildren = 2 * np.sum(topPerformerNonExplorer)

                mutations = np.random.random(N_nonExplorerChildren) > self.explorerRatio

                childrenIndices = np.random.choice(ranking[10:], N_explorerChildren + N_nonExplorerChildren, replace=False)
                alphas[childrenIndices[:N_explorerChildren]] = 0
                alphas[childrenIndices[N_explorerChildren:]] = self.alpha * mutations

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
                

