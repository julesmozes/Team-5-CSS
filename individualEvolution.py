import numpy as np
import minimal.ants as ants
import minimal.osmap as osmap
import matplotlib.pyplot as plt

np.random.seed(49)

class IndividualEvolution:  
    def __init__(
            self, 
            N_ants=100,
            iterations=500,
            generations=10,
            Q=1,
            evaporation=.1,
            fertilityRatio=.1,
            offspringRatio=2,
            mutationRatio=.1
        ):
        
        self.N_ants = N_ants
        self.N_fertile = int(N_ants * fertilityRatio)
        self.N_offspring = int(self.N_fertile * offspringRatio)
        self.N_mutated = int(self.N_offspring * mutationRatio)

        self.Q = 1
        self.evaporation = evaporation

        self.iterations = iterations

        self.alphas = np.empty(N_ants)
        self.betas = np.empty(N_ants)

        self.generations = generations
        self.alphasHistory = np.empty((N_ants, generations))
        self.betasHistory = np.empty((N_ants, generations))
        self.fitnessHistory = np.empty((N_ants, generations))

        return
    
    def makeUniformGenepool(
            self,
            alpha_min=.1,
            alpha_max=5,
            beta_min=.1,
            beta_max=5
        ):
        
        self.alphas = np.random.uniform(alpha_min, alpha_max, self.N_ants)
        self.betas = np.random.uniform(beta_min, beta_max, self.N_ants)

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max

        return
    
    def evolutionStep(self, fitness):
        # replace worst ants with offspring from best
        survivorIndices = np.argsort(fitness)

        bestAnts = survivorIndices[:self.N_fertile]
        worstAnts = survivorIndices[self.N_ants - self.N_offspring:]

        self.alphas[worstAnts] = np.random.choice(self.alphas[bestAnts], self.N_offspring)
        self.betas[worstAnts] = np.random.choice(self.betas[bestAnts], self.N_offspring)

        # apply mutation
        alphasNoise = np.random.uniform(-.1, .1, self.N_mutated)
        betasNoise = np.random.uniform(-1., .1, self.N_mutated)

        mutationIndices = np.random.choice(worstAnts, self.N_mutated, replace=False)

        self.alphas[mutationIndices] += alphasNoise
        self.betas[mutationIndices] += betasNoise

        return
    
    def plot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        alphasMean = np.mean(self.alphasHistory, axis=0) 
        alphasStd = np.std(self.alphasHistory, axis=0)
        betasMean = np.mean(self.betasHistory, axis=0)
        betasStd = np.std(self.betasHistory, axis=0)

        generations = np.arange(self.generations)

        ax1.plot(generations, alphasMean, label="mean")
        ax1.fill_between(
            generations,
            alphasMean - alphasStd,
            alphasMean + alphasStd,
            alpha=.3,
            label="std"
        )

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Alpha value")
        ax1.legend()

        ax2.plot(generations, betasMean, label="mean")
        ax2.fill_between(
            generations,
            betasMean - betasStd,
            betasMean + betasStd,
            alpha=.3,
            label="std"
        )

        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Beta value")
        ax2.legend()

        fitnessMean = np.nanmean(self.fitnessHistory, axis=0)
        fitnessStd = np.nanstd(self.fitnessHistory, axis=0)
        fitnessMin = np.nanmin(self.fitnessHistory, axis=0)

        ax3.plot(generations, fitnessMean, label="mean")
        ax3.plot(generations, fitnessMin, label="min")
        ax3.fill_between(
            generations,
            fitnessMean - fitnessStd,
            fitnessMean + fitnessStd,
            alpha=.3,
            label="std"
        )

        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Fitness value")
        ax3.legend()

        plt.show()

        return
    
    def run(self, min_dist, max_dist):
        map = osmap.Map()
        map.build("Oost, Amsterdam, Netherlands")

        dijkstraDistance = map.makeRandomTest(min_dist, max_dist, plot=False)
        print("minimum distance Dijkstra: ", dijkstraDistance)

        for g in range(self.generations):
            print("--- generation %d ---" % g)
            fitness = np.empty((self.N_ants, self.iterations), dtype=np.float64)

            for it in range(self.iterations):
                solutions = []

                for i in range(self.N_ants):
                    path, length, steps = ants.build_path_numba(
                        map.getNumbaData(),
                        self.alphas[i],
                        self.betas[i],
                        max_steps=1000
                    )

                    if steps:
                        solutions.append((path, length, steps))
                        fitness[i, it] = length
                    else:
                        fitness[i, it] = np.nan

                if not solutions:
                    print(f"iter {it}, no solution")
                    continue

                # ants.update_pheromones_numba(map.pheromone, solutions, self.Q, self.evaporation)
                ants.update_pheromones_numba_max(
                    map.pheromone, 
                    solutions, 
                    self.Q, 
                    self.evaporation, 
                    max_pheromone=.5
                )

                best = min(length for _, length, _ in solutions)
                print(f"iter {it}, best length ratio {best / dijkstraDistance:.3f}")

            fitness /= dijkstraDistance
            fitness = np.nanmean(fitness, axis=1)

            self.alphasHistory[:, g] = self.alphas
            self.betasHistory[:, g] = self.betas
            self.fitnessHistory[:, g] = fitness
            self.evolutionStep(fitness)
            
            map.resetPheromones()

        return
    
evolutionObject = IndividualEvolution(
    N_ants=100, 
    iterations=100, 
    evaporation=.2,
    fertilityRatio=.1,
    offspringRatio=3,
    mutationRatio=.2,
    generations=20
)

evolutionObject.makeUniformGenepool(
    alpha_min = -5,
    alpha_max = 10,
    beta_min = -5,
    beta_max = 10
)

evolutionObject.run(800, 1000)

evolutionObject.plot()
                

