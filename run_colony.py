from colony import Colony

distances = [
    [0, 2, 2, 5, 7],
    [2, 0, 4, 8, 2],
    [2, 4, 0, 1, 3],
    [5, 8, 1, 0, 2],
    [7, 2, 3, 2, 0],
]

colony = Colony(
    num_ants=5,
    num_nodes=5,
    distances=distances,
    alpha=1,
    beta=2,
    evaporation_rate=0.5,
    Q=100,
    start_node=0
)

best_tour, best_cost = colony.run(iterations=50)

print("\nBest Tour:", best_tour)
print("Best Cost:", best_cost)
