import numpy as np
report = np.load("compare_evo_vs_grid_2026-01-29T13-22-01.npz", allow_pickle=True)
print(report["report"])