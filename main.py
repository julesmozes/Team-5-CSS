import numpy as np
import matplotlib.pyplot as plt
import networkx

class Edge:
    def __init__(self, start, end, cost=np.random.rand() + .1):
        self.start = start
        self.end = end

        self.cost = cost
        self.pheromone = 0
        self.deposited = 0

        return
    
    def update(self):
        self.pheromone = (1 - evaporation) * self.pheromone + self.deposited
        self.deposited = 0

        return
    
N_network = 50
p = .2

G = networkx.erdos_renyi_graph(N_network, p)

networkx.draw(G)
plt.show()

edges = [Edge(x[0], x[1]) for x in G.edges()]

edgesStart = [[] for _ in range(N_network)]
for edge in edges:
    edgesStart[edge.start].append(edge)
    edgesStart[edge.end].append(edge)
    

class Ant:
    def __init__(self, start, dest, alpha, beta):
        self.start = start
        self.pos = start
        self.dest = dest

        self.alpha = alpha
        self.beta = beta

        self.path = []
        self.pathLen = 0
        self.arrived = False

        return

    def p(self, edge):
        return edge.pheromone ** self.alpha * edge.cost ** self.beta

    def step(self):
        if self.arrived:
            return
        
        possibilities = edgesStart[self.pos]

        probabilities = []
        for i in range(len(possibilities)):
            probabilities.append(self.p(possibilities[i]))

        totP = sum(probabilities)
        if totP:
            for i in range(len(probabilities)):
                probabilities[i] /= totP

            chosenEdge = np.random.choice(possibilities, p=probabilities)

        # edge case: when no pheromone yet -> all p are zero
        else:
            chosenEdge = np.random.choice(possibilities)

        if chosenEdge.start == self.pos:
            self.pos = chosenEdge.end
        else:
            self.pos = chosenEdge.start

        self.path.append(chosenEdge)

        if self.pos == self.dest:
            self.arrived = True

        return
    
    def update(self):
        self.pathLen = 0
        for edge in self.path:
            self.pathLen += edge.cost

        for edge in self.path:
            edge.deposited += Q / self.pathLen

        self.path = []
        self.arrived = False
        self.pos = self.start

        return

N_ants = 20
maxIterAnt = int(1e4)
maxIterColony = int(1e2)

threshold = .8
threshold = int(threshold * N_ants)

Q = 10
alpha = 2
beta = -1.5
evaporation = .3

startEdge, endEdge = np.random.choice(edges, 2, replace=False)
start = startEdge.start
end = endEdge.end

ants = [Ant(start=start, dest=end, alpha=alpha, beta=beta) for _ in range(N_ants)]

for i in range(maxIterColony):
    for j in range(maxIterAnt):
        done = True
        for ant in ants:
            ant.step() 
            done = done and ant.arrived

        if done:
            break
    
    for ant in ants:
        ant.update()

    pathLens = [ant.pathLen for ant in ants]
    pathLens.sort()

    j = 0
    while j < N_ants and pathLens[j] == pathLens[0]:
        j += 1
    
    if (j >= threshold):
        print("total iterations: %d" % i)
        break

    for edge in edges:
        edge.update()


for edge in edges:
    print(edge.pheromone)


# notes

# make graph directed, also higher p
# make random path for initial condition
# fix decision function
# reset ant position


# optimisations

# dont store entire path -> extrapolate from pheromones
# precompute decision function for edges -> in Edge.update()



