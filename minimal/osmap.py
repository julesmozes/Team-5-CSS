import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import copy

import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib import colors

class Map:
    def __init__(self):
        self.G = None
        self.src = None
        self.dest = None

    def largest_strongly_connected_component(self, G):
        sccs = nx.strongly_connected_components(G)
        largest = max(sccs, key=len)
        return G.subgraph(largest).copy()

    def multidigraph_to_digraph(self, Gm, weight="cost") -> nx.DiGraph:
        G = nx.DiGraph()

        # copy nodes
        G.add_nodes_from(Gm.nodes(data=True))

        for u, v, data in Gm.edges(data=True):
            if G.has_edge(u, v):
                # keep the cheaper edge
                if data[weight] < G[u][v][weight]:
                    G[u][v].update(data)
            else:
                G.add_edge(u, v, **data)

        return G

    def build(self, place: str, network_type: str = "drive", add_travel_time: bool = True) -> nx.DiGraph:
        self.Gm = ox.graph_from_place(place, network_type=network_type, simplify=True)

        # Ensure we have edge lengths (usually already present)
        if not any("length" in data for *_, data in self.Gm.edges(keys=True, data=True)):
            self.Gm = ox.distance.add_edge_lengths(self.Gm)

        # Optional: infer speeds + travel times
        if add_travel_time:
            self.Gm = ox.routing.add_edge_speeds(self.Gm)
            self.Gm = ox.routing.add_edge_travel_times(self.Gm)

        # Add dynamic state fields
        for u, v, k, data in self.Gm.edges(keys=True, data=True):
            data.setdefault("open", True)

            base = float(data["travel_time"]) if add_travel_time and "travel_time" in data else float(data["length"])
            data["cost"] = base
            data["base_cost"] = base
            data["pheromone"] = 1e-4
            data["heuristic"] = 1.0

        
        self.Gm = self.largest_strongly_connected_component(self.Gm)
        self.Gm = ox.project_graph(self.Gm)
        self.G = self.multidigraph_to_digraph(self.Gm) 
        self.graph_to_arrays(self.G)

        return

    def random_node_pair(self, min_dist, max_dist, max_tries=10000):

        for _ in range(max_tries):
            self.src, self.dest = np.random.choice(np.arange(self.N), 2, replace=False)

            self.src_node, self.dest_node = self.nodes[self.src], self.nodes[self.dest]

            # Euclidean separation (cheap pre-filter)
            x1, y1 = self.G.nodes[self.src_node]["x"], self.G.nodes[self.src_node]["y"]
            x2, y2 = self.G.nodes[self.dest_node]["x"], self.G.nodes[self.dest_node]["y"]
            eucl = np.hypot(x1 - x2, y1 - y2)
            if eucl < min_dist:
                continue

            # Directed network distance (true difficulty)
            try:
                d = nx.shortest_path_length(self.G, self.src_node, self.dest_node, weight="cost")
            except nx.NetworkXNoPath:
                continue

            if d <= max_dist:
                return d

        raise RuntimeError("Could not find suitable node pair within distance bounds")

    def build_heuristic(self):
        target_x, target_y = self.G.nodes[self.dest_node]["x"], self.G.nodes[self.dest_node]["y"]

        for i in range(self.N):
            for k in range(self.degree[i]):
                posIndex = self.neighbors[i, k]
                pos = self.nodes[posIndex]
                x, y = self.G.nodes[pos]["x"], self.G.nodes[pos]["y"]
                self.heuristic[i, k] = 1 / (np.hypot(x - target_x, y - target_y) + 1e-6)

        return
    
    def plot(self):
        fig, ax = ox.plot_graph(
            self.Gm,
            node_size=0,
            edge_linewidth=0.6,
            bgcolor="white",
            show=False,
            close=False
        )

        if self.src:
            ax.scatter(
                self.G.nodes[self.src_node]["x"],
                self.G.nodes[self.src_node]["y"],
                c="green",
                s=80,
                zorder=5,
                label="source",
            )

        if self.dest:
            ax.scatter(
                self.G.nodes[self.dest_node]["x"],
                self.G.nodes[self.dest_node]["y"],
                c="red",
                s=80,
                zorder=5,
                label="target",
            )

        ax.legend()
        plt.show()

        return
    
    def animate_costs(
    self,
    t_max=200,
    dt=1.0,
    interval=50,
    cmap="rainbow"
    ):
        """
        Animate dynamic edge costs.
        
        Parameters
        ----------
        t_max : float
            Maximum simulation time
        dt : float
            Time step per frame
        interval : int
            Delay between frames in ms
        cmap : str
            Matplotlib colormap
        """

        # --- build edge geometry once ---
        segments = []
        edge_index = []  # (i, k) pairs

        for i, u in enumerate(self.nodes):
            x1, y1 = self.G.nodes[u]["x"], self.G.nodes[u]["y"]
            for k in range(self.degree[i]):
                j = self.neighbors[i, k]
                v = self.nodes[j]
                x2, y2 = self.G.nodes[v]["x"], self.G.nodes[v]["y"]

                segments.append([(x1, y1), (x2, y2)])
                edge_index.append((i, k))

        segments = np.asarray(segments)

        # --- initial colors ---
        base = np.array([self.base_cost[i, k] for i, k in edge_index])
        cost = np.array([self.cost[i, k] for i, k in edge_index])
        ratio = cost / base

        norm = colors.Normalize(vmin=0.5, vmax=1.5)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=1.0
        )
        lc.set_array(ratio)

        # --- plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_axis_off()

        cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("cost / base_cost")

        def update(frame):
            t = frame * dt
            self.updateEdges(t)

            cost = np.fromiter(
                (self.cost[i, k] for i, k in edge_index),
                dtype=float,
                count=len(edge_index)
            )

            lc.set_array(cost / base)
            ax.set_title(f"t = {t:.1f}")

            return ()

        frames = int(t_max / dt)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=False,
            repeat=True
        )

        plt.show()
        return
    
    def graph_to_arrays(self, G):
        self.N = len(G.nodes)

        self.max_deg = max(len(list(G.neighbors(n))) for n in G.nodes())

        self.nodes      = np.array([u for u in G.nodes])
        self.pos        = np.zeros((self.N, 2), dtype=np.float64)
        self.neighbors  = np.ones((self.N, self.max_deg), dtype=np.int32)
        self.pheromone  = np.ones((self.N, self.max_deg), dtype=np.float64) * .1
        self.heuristic  = np.zeros((self.N, self.max_deg), dtype=np.float64)
        self.base_cost  = np.zeros((self.N, self.max_deg), dtype=np.float64)
        self.cost       = np.zeros((self.N, self.max_deg), dtype=np.float64)
        self.modulation = np.zeros(self.N, dtype=np.float64)
        self.degree     = np.zeros(self.N, dtype=np.int32)

        for i, u in enumerate(G.nodes):
            self.pos[i, :] = [G.nodes[u]["x"], G.nodes[u]["y"]]

            for k, v in enumerate(G.neighbors(u)):
                self.neighbors[i, k] = np.argwhere(self.nodes == v)
                edge = G[u][v]
                self.pheromone[i, k] = edge["pheromone"]
                self.heuristic[i, k] = edge["heuristic"]
                self.base_cost[i, k] = edge["cost"]
                self.cost[i, k]      = edge["cost"]
                self.degree[i] += 1

        return
    
    def calcModulation(self, t):
        for i in range(self.N):
            self.modulation[i] = self.amplitude @ np.sin(self.waveVectors @ self.pos[i, :] + self.tempFrequencies * t)

        return
    
    def buildTraffic(
        self, 
        modulationDepth=.5,
        minSpatialWavelength=200,
        minTemporalWavelength=5,
        maxTime=100,
        steps=100
    ):
        # use 10 components for spatial/temporal
        self.waveVectors = np.random.uniform(-1, 1, (10, 2)) / minSpatialWavelength
        self.tempFrequencies = np.random.uniform(-1, 1, 10) / minTemporalWavelength
        self.amplitude = np.random.uniform(-1, 1, 10)

        # normalise
        time = np.linspace(0, maxTime, steps)
        maxModulation = np.empty(steps)

        for i, t in enumerate(time):
            self.calcModulation(t)

            edgeModulation = np.zeros((self.N, self.max_deg))
            for u in range(self.N):
                for k in range(self.degree[u]):
                    v = self.neighbors[u, k]

                    edgeModulation[u, k] = .5 * (self.modulation[u] + self.modulation[v])

            maxModulation[i] = np.max(np.abs(edgeModulation))

        self.amplitude *= modulationDepth / np.max(maxModulation)

        return
    
    def updateEdges(self, t):
        self.calcModulation(t)

        for i in range(self.N):
            for k in range(self.degree[i]):
                j = self.neighbors[i, k]

                self.cost[i, k] = self.base_cost[i, k] * np.exp(.5 * (self.modulation[i] + self.modulation[j]))

        for i, u in enumerate(self.G.nodes):
            for k, v in enumerate(self.G.neighbors(u)):
                j = np.argwhere(self.nodes == v)
                edge = self.G[u][v]
                base_cost = edge["base_cost"]
                edge["cost"] = base_cost * np.exp(.5 * (self.modulation[i] + self.modulation[j]))

        return
    
    def getShortestPathLength(self):
        return float(
            nx.shortest_path_length(
                self.G,
                self.src_node,
                self.dest_node,
                weight="cost"
            )
        )
    
    def seedShortestPath(self, pheromone=1):
        self.optimalPath = nx.dijkstra_path(self.G, self.src_node, self.dest_node, weight="cost")

        for i in range(len(self.optimalPath) - 1):
            current = np.argwhere(self.nodes == self.optimalPath[i])[0][0]
            options = self.nodes[self.neighbors[current, :self.degree[current]]]
            k = np.argwhere(options == self.optimalPath[i + 1])

            self.pheromone[current, k] = pheromone

        return
    
    def evaluateBaselinePath(self):
        length = 0

        for i in range(len(self.optimalPath) - 1):
            current = np.argwhere(self.nodes == self.optimalPath[i])[0][0]
            options = self.nodes[self.neighbors[current, :self.degree[current]]]
            k = np.argwhere(options == self.optimalPath[i + 1])

            length += self.cost[current, k]

        return length

    def resetPheromones(self):
        for u, v in self.G.edges:
            self.G[u][v]["pheromone"] = .1

        for i in range(self.N):
            self.pheromone[i] = np.ones(self.max_deg) * .1

        return
    
    def makeRandomTest(self, min_dist=100, max_dist=300, plot=True, resetPheromones=True):
        d = self.random_node_pair(min_dist, max_dist)
        self.build_heuristic()

        if resetPheromones:
            self.resetPheromones()

        if plot:
            self.plot()

        return d
    
    def getNumbaData(self):
        return (self.max_deg, 
                self.neighbors,
                self.degree,
                self.heuristic,
                self.pheromone,
                self.cost,
                self.src,
                self.dest,)
    
    def getNumbaDataDynamic(self):
        return (self.max_deg, 
                self.neighbors,
                self.degree,
                self.heuristic,
                self.pheromone,
                self.base_cost,
                self.pos,
                self.waveVectors,
                self.tempFrequencies,
                self.amplitude,
                self.src,
                self.dest,)

    def copy(self):
        return copy.deepcopy(self)