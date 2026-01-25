import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

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

        return

    def random_node_pair(self, min_dist, max_dist, max_tries=10000):
        nodes = list(self.G.nodes)

        for _ in range(max_tries):
            self.src = np.random.choice(nodes)
            self.dest = np.random.choice(nodes)
            if self.src == self.dest:
                continue

            # Euclidean separation (cheap pre-filter)
            x1, y1 = self.G.nodes[self.src]["x"], self.G.nodes[self.src]["y"]
            x2, y2 = self.G.nodes[self.dest]["x"], self.G.nodes[self.dest]["y"]
            eucl = np.hypot(x1 - x2, y1 - y2)
            if eucl < min_dist:
                continue

            # Directed network distance (true difficulty)
            try:
                d = nx.shortest_path_length(self.G, self.src, self.dest, weight="cost")
            except nx.NetworkXNoPath:
                continue

            if d <= max_dist:
                return self.src, self.dest

        raise RuntimeError("Could not find suitable node pair within distance bounds")

    def build_heuristic(self):
        target_x, target_y = self.G.nodes[self.dest]["x"], self.G.nodes[self.dest]["y"]

        for u, v in self.G.edges:
            x, y = self.G.nodes[v]["x"], self.G.nodes[v]["y"]
            self.G[u][v]["heuristic"] = 1 / (np.hypot(x - target_x, y - target_y) + 1e-6)

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
                self.G.nodes[self.src]["x"],
                self.G.nodes[self.src]["y"],
                c="green",
                s=80,
                zorder=5,
                label="source",
            )

        if self.dest:
            ax.scatter(
                self.G.nodes[self.dest]["x"],
                self.G.nodes[self.dest]["y"],
                c="red",
                s=80,
                zorder=5,
                label="target",
            )

        ax.legend()
        plt.show()

        return
    
    def makeRandomTest(self, min_dist=100, max_dist=300, plot=True, resetPheromones=True):
        self.random_node_pair(min_dist, max_dist)
        self.build_heuristic()

        if resetPheromones:
            for u, v in self.G.edges:
                self.G[u][v]["pheromone"] = 1.0

        if plot:
            self.plot()

        return nx.shortest_path_length(self.G, self.src, self.dest, weight="cost")
