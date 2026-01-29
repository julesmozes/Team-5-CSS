import numpy as np
import osmap 
import ants

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib import colors

np.random.seed(44)

map = osmap.Map()

map.build("Oost, Amsterdam, Netherlands")
min_dist = map.makeRandomTest(500, 800, plot=False)

# map.seedShortestPath()

print("minimum distance Dijkstra: ", min_dist)

def transparent_cmap(color, name="transparent_cmap", N=256):
    alphas = np.linspace(0, 1, N)

    r, g, b = color

    colormap = np.zeros((N, 4))
    colormap[:, 0] = r
    colormap[:, 1] = g
    colormap[:, 2] = b
    colormap[:, 3] = alphas

    return colors.LinearSegmentedColormap.from_list(name, colormap)

redMap = transparent_cmap((1, 0, 0))

N_ANTS = 500
ITER = 500
alpha = 1
beta = 1.5
Q = 1
evaporation = .2
max_pheromone = .2

segments = []
edge_index = []  # (i, k) pairs

for i, u in enumerate(map.nodes):
    x1, y1 = map.G.nodes[u]["x"], map.G.nodes[u]["y"]
    for k in range(map.degree[i]):
        j = map.neighbors[i, k]
        v = map.nodes[j]
        x2, y2 = map.G.nodes[v]["x"], map.G.nodes[v]["y"]

        segments.append([(x1, y1), (x2, y2)])
        edge_index.append((i, k))

segments = np.asarray(segments)

norm = colors.Normalize(vmin=0, vmax=max_pheromone)

color = np.array([map.pheromone[i, k] for i, k in edge_index])

lcBase = LineCollection(
    segments,
    color="grey",
    linewidths=1.0,
    zorder=1,
    alpha=0.5
)

lc = LineCollection(
    segments,
    cmap=redMap,
    norm=norm,
    linewidths=1.0
)
lc.set_array(color)

fig, ax = plt.subplots(figsize=(8, 8))
ax.add_collection(lc)
ax.add_collection(lcBase)
ax.autoscale()
ax.set_axis_off()

cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("Pheromone")

ax.scatter(
    map.G.nodes[map.src_node]["x"],
    map.G.nodes[map.src_node]["y"],
    c="green",
    s=80,
    zorder=5,
    label="source",
)

ax.scatter(
    map.G.nodes[map.dest_node]["x"],
    map.G.nodes[map.dest_node]["y"],
    c="red",
    s=80,
    zorder=5,
    label="target",
)

def update(frame):
    solutions = []

    for _ in range(N_ANTS):
        path, length, steps = ants.build_path_numba(map.getNumbaData(), alpha, beta, max_steps=1000)
        if steps:
            solutions.append((path, length, steps))

    if solutions:
        ants.update_pheromones_numba_max(map.pheromone, solutions, Q, evaporation, max_pheromone=max_pheromone)
        color = np.array([map.pheromone[i, k] for i, k in edge_index])
        lc.set_array(color)

    ax.set_title(f"iteration = {frame:.1f}")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=ITER,
    interval=100,
    blit=False,
    repeat=False
)

plt.show()