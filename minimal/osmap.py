import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import copy

class Map:
    def __init__(self):
        self.G = None
        self.src = None
        self.dst = None

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
            data["pheromone"] = 1.0
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
    
    def graph_to_arrays(self, G):
        self.N = len(G.nodes)

        self.max_deg = max(len(list(G.neighbors(n))) for n in G.nodes())

        self.nodes      = np.array([u for u in G.nodes])
        self.neighbors  = np.ones((self.N, self.max_deg), dtype=np.int32)
        self.pheromone  = np.ones((self.N, self.max_deg), dtype=np.float64)
        self.heuristic  = np.zeros((self.N, self.max_deg), dtype=np.float64)
        self.cost       = np.zeros((self.N, self.max_deg), dtype=np.float64)
        self.degree     = np.zeros(self.N, dtype=np.int32)

        for i, u in enumerate(G.nodes):
            for k, v in enumerate(G.neighbors(u)):
                self.neighbors[i, k] = np.argwhere(self.nodes == v)
                edge = G[u][v]
                self.pheromone[i, k] = edge["pheromone"]
                self.heuristic[i, k] = edge["heuristic"]
                self.cost[i, k]      = edge["cost"]
                self.degree[i] += 1

        return
    
    def resetPheromones(self):
        for u, v in self.G.edges:
            self.G[u][v]["pheromone"] = 1.0

        for i in range(self.N):
            self.pheromone[i] = np.ones(self.max_deg)

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

    def copy(self):
        return copy.deepcopy(self)