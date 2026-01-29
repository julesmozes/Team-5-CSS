import numpy as np
import osmap 
import ants

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib import colors
from matplotlib.animation import PillowWriter

np.random.seed(49)

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

redMap = transparent_cmap((0, 1, 0))

N_ANTS = 500
ITER = 500
alpha = 1
beta = 1.5
Q = 1
evaporation = .2
max_pheromone = 2

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
ax.set_axis_off()

fig.patch.set_facecolor("black")
ax.set_facecolor("black")

sx, sy = map.G.nodes[map.src_node]["x"], map.G.nodes[map.src_node]["y"]
dx, dy = map.G.nodes[map.dest_node]["x"], map.G.nodes[map.dest_node]["y"]

xmin = min(sx, dx)
xmax = max(sx, dx)
ymin = min(sy, dy)
ymax = max(sy, dy)

xc = .5 * (xmin + xmax)
yc = .5 * (ymin + ymax)

radius = 0.8 * max(abs(xmax - xmin), abs(ymax - ymin))

ax.set_xlim(xc - radius, xc + radius)
ax.set_ylim(yc - radius, yc + radius)

cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("Pheromone", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.get_yticklabels(), color="white")

ax.scatter(
    map.G.nodes[map.src_node]["x"],
    map.G.nodes[map.src_node]["y"],
    c="white",
    s=80,
    zorder=5,
    label="source",
)

ax.scatter(
    map.G.nodes[map.dest_node]["x"],
    map.G.nodes[map.dest_node]["y"],
    c="white",
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

    ax.set_title(f"iteration = {frame:.1f}", color="white")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=ITER,
    interval=100,
    blit=False,
    repeat=False
)

gif_path = "ants_pheromones.gif"

writer = PillowWriter(
    fps=20,          # frames per second
    metadata={"artist": "Ant Colony Optimization"},
    bitrate=1800
)

ani.save(gif_path, writer=writer)

plt.show()