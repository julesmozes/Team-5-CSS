import numpy as np
import matplotlib.pyplot as plt
import networkx

N_network = 50
p = .2

G = networkx.erdos_renyi_graph(N_network, p)

networkx.draw(G)
plt.show()

# Initialize per-edge attributes for ACO
# cost: static edge weight (can later become dynamic)
# pheromone: trail strength
# deposited: per-iteration accumulator (applied during global update)
pheromone_init = 1.0
for u, v in G.edges():
    G[u][v]["cost"] = float(np.random.rand() + 0.1)
    G[u][v]["pheromone"] = float(pheromone_init)
    G[u][v]["deposited"] = 0.0
    

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

    def p(self, edge_data):
        pher = edge_data.get("pheromone", 0.0)
        cost = edge_data.get("cost", 1.0)
        return (pher ** self.alpha) * (cost ** self.beta)

    def step(self, G):
        if self.arrived:
            return

        # Candidate edges incident to current position
        possibilities = list(G.edges(self.pos, data=True))  # (u, v, data)
        if not possibilities:
            return

        # Compute probabilities
        weights = np.array([self.p(data) for (_, _, data) in possibilities], dtype=float)
        tot = float(weights.sum())

        if tot > 0:
            probs = weights / tot
            idx = int(np.random.choice(len(possibilities), p=probs))
        else:
            idx = int(np.random.choice(len(possibilities)))

        u, v, _ = possibilities[idx]
        nxt = v if u == self.pos else u

        # Store the traversed edge as an ordered tuple so we can update it later
        self.path.append((u, v))
        self.pos = nxt

        if self.pos == self.dest:
            self.arrived = True
    
    def update(self, G, Q):
        # Compute path length
        self.pathLen = 0.0
        for u, v in self.path:
            self.pathLen += float(G[u][v].get("cost", 1.0))

        # Deposit pheromone along the path
        if self.pathLen > 0:
            deposit_amt = float(Q) / float(self.pathLen)
            for u, v in self.path:
                G[u][v]["deposited"] = float(G[u][v].get("deposited", 0.0)) + deposit_amt

        # Reset ant for next iteration
        self.path = []
        self.arrived = False
        self.pos = self.start


class Simulate:
    def __init__(
        self,
        G,
        start,
        end,
        n_ants=20,
        max_iter_ant=int(1e4),
        max_iter_colony=int(1e2),
        threshold_frac=0.8,
        Q=10.0,
        alpha=2.0,
        beta=-1.5,
        evaporation=0.3,
        seed=None,
    ):
        self.G = G
        self.start = start
        self.end = end
        self.n_ants = int(n_ants)
        self.max_iter_ant = int(max_iter_ant)
        self.max_iter_colony = int(max_iter_colony)
        self.threshold = int(threshold_frac * self.n_ants)

        self.Q = float(Q)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaporation = float(evaporation)

        if seed is not None:
            np.random.seed(int(seed))

        self.ants = [Ant(start=start, dest=end, alpha=self.alpha, beta=self.beta) for _ in range(self.n_ants)]

    def global_pheromone_update(self):
        # Apply evaporation and add the per-iteration deposited pheromone
        for u, v, data in self.G.edges(data=True):
            pher = float(data.get("pheromone", 0.0))
            dep = float(data.get("deposited", 0.0))
            data["pheromone"] = (1.0 - self.evaporation) * pher + dep
            data["deposited"] = 0.0

    def run(self):
        for i in range(self.max_iter_colony):
            # Let ants walk until all have arrived or we hit the per-ant step cap
            for _ in range(self.max_iter_ant):
                done = True
                for ant in self.ants:
                    ant.step(self.G)
                    done = done and ant.arrived
                if done:
                    break

            # Deposit pheromone based on paths found this iteration
            for ant in self.ants:
                ant.update(self.G, self.Q)

            # Check convergence: how many ants achieved the current best path length?
            path_lens = sorted([ant.pathLen for ant in self.ants])
            j = 0
            while j < self.n_ants and path_lens[j] == path_lens[0]:
                j += 1

            if j >= self.threshold:
                print(f"total iterations: {i}")
                break

            # Update pheromones for next iteration
            self.global_pheromone_update()

        return i


N_ants = 20
maxIterAnt = int(1e4)
maxIterColony = int(1e2)

threshold_frac = 0.8

Q = 10
alpha = 2
beta = -1.5
evaporation = 0.3

# Pick random start/end nodes from the graph
nodes = list(G.nodes())
start, end = np.random.choice(nodes, 2, replace=False)

sim = Simulate(
    G=G,
    start=start,
    end=end,
    n_ants=N_ants,
    max_iter_ant=maxIterAnt,
    max_iter_colony=maxIterColony,
    threshold_frac=threshold_frac,
    Q=Q,
    alpha=alpha,
    beta=beta,
    evaporation=evaporation,
)

sim.run()

# Print final pheromones (debug)
for u, v, data in G.edges(data=True):
    print(data.get("pheromone", 0.0))
