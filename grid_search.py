import minimal.ants as ants
import minimal.osmap as osmap
import numpy as np
import networkx as nx

# map = osmap.Map()

# map.build("Oost, Amsterdam, Netherlands")
# min_dist = map.makeRandomTest(300, 500, plot=False)

# print("minimum distance Dijkstra: ", min_dist)


class Grid_search():
    def __init__(self, n_ants, iterations, Q, evaporation, init_params, repeats=5, base_seed=67):
        self.n_ants = n_ants
        self.iterations = iterations
        self.Q = Q
        self.evaporation = evaporation
        self.alphas = init_params[0]
        self.betas = init_params[1]
        self.repeats = repeats
        self.base_seed = base_seed
        
    def run_manual_grid_search(self):
        results = {}
        
        for a in self.alphas:
            for b in self.betas:
                results[(a, b)] = self.eval_one(a, b)
                
        return results 
    
    def eval_one(self, a, b):
        target_dist = self.min_dist + 50

        best_lengths = []
        mean_norms = []
        solved_iterations = []

        for r in range(self.repeats):
            solved = False
            np.random.seed(self.base_seed + r)
            self.map.resetPheromones()

            iters_solved = self.iterations
            best_length = float("inf")
            all_lengths = []

            for it in range(self.iterations):
                solutions = []

                for i in range(self.n_ants):
                    path, length, steps = ants.build_path_numba(
                        self.map.getNumbaData(), a, b, max_steps=1000
                    )
                    if steps:
                        solutions.append((path, length, steps))
                        all_lengths.append(length)

                if not solutions:
                    continue

                ants.update_pheromones_numba_max(
                    self.map.pheromone, solutions, self.Q, self.evaporation
                )

                current_best = min(length for _, length, _ in solutions)
                if current_best < best_length:
                    best_length = current_best

                if not solved and current_best <= target_dist:
                    iters_solved = it
                    solved = True

            solved_iterations.append(iters_solved)
            best_lengths.append(best_length)

            mean_len = np.mean(all_lengths) if all_lengths else float("inf")
            mean_norms.append(mean_len / self.min_dist)

        return {
                "fitness": float(np.mean(solved_iterations)),
                "best_length": float(np.mean(best_lengths)),
                "mean_norm": float(np.mean(mean_norms)),
                }     
        
    def find_srcdest(self, filename):
        self.evo_data = np.load(filename, allow_pickle=True)
        best_paths = self.evo_data["best_paths"]
        fitness_hist = self.evo_data["fitness_history"]
        flat_fitness = fitness_hist[:, :, 0]

        last_gen_idx = -1
        scores_last_gen = flat_fitness[last_gen_idx]
        best_colony_idx = np.argmin(scores_last_gen)
        winning_path = best_paths[last_gen_idx][best_colony_idx]

        start_node_id = winning_path[0]
        end_node_id = winning_path[-1]
        
        return start_node_id, end_node_id
    
    def build_map(self, filename=None, random_=False):
        self.map = osmap.Map()
        self.map.build("Amsterdam, Netherlands")
        
        if not random_:
            self.map.src_node, self.map.dest_node = self.find_srcdest(filename)
            
            try:
                self.map.src = int(np.where(self.map.nodes == self.map.src_node)[0][0])
                self.map.dest = int(np.where(self.map.nodes == self.map.dest_node)[0][0])
            
            except IndexError:
                raise ValueError("Saved nodes not found in current map. Chekc city strings")
            
            self.map.build_heuristic()
            self.min_dist = nx.shortest_path_length(self.map.G, self.map.src_node, self.map.dest_node, weight="cost")
        
        else:
            self.min_dist = self.map.makeRandomTest(300, 500, plot=False)
            
    def pick_best(self, results):
        return min(results, key=lambda p: (results[p]["fitness"], results[p]["mean_norm"]))
    
    def make_fine_grid(self, a_star, b_star, step_a=0.25, step_b=0.25):
        alphas = [round(a_star - step_a, 3), round(a_star, 3), round(a_star + step_a, 3)]
        betas = [round(b_star - step_b, 3), round(b_star, 3), round(b_star + step_b, 3)]
        
        alphas = [a for a in alphas if a > 0]
        betas = [b for b in betas if b > 0]
        return alphas, betas
    
    def run_coarse_to_fine(self, fine_step_a=0.25, fine_step_b=0.25, mode="manual"):
        if mode == "manual":
            results_coarse = self.run_manual_grid_search()
            best_a, best_b = self.pick_best(results_coarse)
            best_metrics = results_coarse[(best_a, best_b)]

        elif mode == "random":
            (best_a, best_b), best_metrics = self.run_random_search()
            results_coarse = {(best_a, best_b): best_metrics}

        else:
            raise ValueError("mode must be 'manual' or 'random' :((")

        print("\nBEST coarse:", best_a, best_b, best_metrics)

        fine_alphas, fine_betas = self.make_fine_grid(best_a, best_b, fine_step_a, fine_step_b)

        old_alphas, old_betas = self.alphas, self.betas
        self.alphas, self.betas = fine_alphas, fine_betas

        old_iters, old_repeats = self.iterations, self.repeats
        self.iterations = max(self.iterations, 400)
        self.repeats = max(self.repeats, 3)

        results_fine = self.run_manual_grid_search()
        best_a2, best_b2 = self.pick_best(results_fine)

        print("\nBEST fine:", best_a2, best_b2, results_fine[(best_a2, best_b2)])

        self.alphas, self.betas = old_alphas, old_betas
        self.iterations, self.repeats = old_iters, old_repeats

        return results_coarse, results_fine, (best_a2, best_b2)
    
    def run_random_search(self, n_samples=30, alpha_range=(0.5, 5.0), beta_range=(0.75, 3.0)):
        rng = np.random.default_rng(self.base_seed)

        best_params = None
        best_metrics = None

        for _ in range(n_samples):
            a = float(rng.uniform(*alpha_range))
            b = float(rng.uniform(*beta_range))
            
            print(f"Now trying alpha={a:.3f}, beta={b:.3f}")

            metrics = self.eval_one(a, b)

            if best_metrics is None:
                best_params, best_metrics = (a, b), metrics
            else:
                if (metrics["fitness"], metrics["mean_norm"]) < (best_metrics["fitness"], best_metrics["mean_norm"]):
                    best_params, best_metrics = (a, b), metrics

            print(f"BOEMBASTIC METRICS:\n fitness={metrics['fitness']:.2f}, mean_norm={metrics['mean_norm']:.4f}")

        print("\nBEST random:", best_params, best_metrics)
        return best_params, best_metrics
                
        
        
if __name__ == "__main__":
    alphas = [0.1, 1, 2, 3, 4, 5]
    betas  = [1, 2, 3, 4, 5, 6]
    init_params = (alphas, betas)
    
    n_ants = 200
    iterations = 1000
    alpha = 2
    beta = 1.5
    Q = 1
    evaporation = 0.2
    
    grid_search = Grid_search(n_ants, iterations, Q, evaporation, init_params)
    grid_search.build_map(filename="new_version_colony_evolution/evolution_backup.npz")
    grid_search.run_random_search(n_samples=1)
    
    




        
# def run_grid_search(self):
#         results = {}
        
#         target_dist = self.min_dist + 50
        
#         for a in self.alphas:
#             for b in self.betas:
#                 best_lengths = []
#                 mean_norms = []
#                 solved_iterations = []
                
                
#                 for r in range(self.repeats):
#                     solved = False
#                     np.random.seed(self.base_seed + r)
#                     self.map.resetPheromones()
                    
#                     iters_solved = self.iterations
#                     best_length = float("inf")
#                     all_lengths = []
                    
                    
#                     for it in range(self.iterations):
#                         solutions = []
                        
#                         for i in range(self.n_ants):
#                             path, length, steps = ants.build_path_numba(self.map.getNumbaData(), a, b, max_steps=1000)
#                             if steps:
#                                 solutions.append((path, length, steps))
#                                 all_lengths.append(length)
                                
#                         if not solutions:
#                             continue
                        
#                         ants.update_pheromones_numba_max(self.map.pheromone, solutions, self.Q, self.evaporation, max_pheromone=0.5)
                        
#                         current_best = min(length for _, length, _ in solutions)
#                         if current_best < best_length:
#                             best_length = current_best
                        
#                         if not solved and current_best <= target_dist:
#                             iters_solved = it
#                             solved = True
                    
#                     solved_iterations.append(iters_solved)  
#                     best_lengths.append(best_length)
                    
#                     mean_len = np.mean(all_lengths) if all_lengths else float('inf')
#                     mean_norms.append(mean_len / self.min_dist)
                    
#                 mean_fitness = float(np.mean(solved_iterations))

#                 results[(a, b)] = {
#                 "fitness": mean_fitness,
#                 "best_length": float(np.mean(best_lengths)),
#                 "mean_norm": float(np.mean(mean_norms)),
#                 }
                
#         return results 




# def run_grid_search(alphas, betas, N_ants, iterations, Q, evaporation, map, repeats=5, base_seed=67):
#     results = {}
#     for alpha in alphas:
#         for beta in betas:
#             best_lengths = []
#             mean_norms = []

#             for r in range(repeats):
#                 np.random.seed(base_seed + r)
#                 map.resetPheromones()

#                 best_length = float('inf')
#                 all_lengths = []

#                 for it in range(iterations):
#                     solutions = []

#                     for i in range(N_ants):
#                         path, length, steps = ants.build_path_numba(map.getNumbaData(), alpha, beta, max_steps=1000)
#                         if steps:
#                             solutions.append((path, length, steps))
#                             all_lengths.append(length)

#                     if not solutions:
#                         continue

#                     ants.update_pheromones_numba_max(map.pheromone, solutions, Q, evaporation, max_pheromone=0.5)

#                     current_best = min(length for _, length, _ in solutions)
#                     if current_best < best_length:
#                         best_length = current_best

#                 best_lengths.append(best_length)
                
#                 mean_len = np.mean(all_lengths) if all_lengths else float('inf')
#                 mean_norms.append(mean_len / min_dist)

#             results[(alpha, beta)] = {
#                 "best_length": float(np.mean(best_lengths)),
#                 "mean_norm": float(np.mean(mean_norms)),
#             }
#             print(
#                 f"alpha: {alpha}, beta: {beta}, mean_norm: {results[(alpha, beta)]['mean_norm']:.4f}, "
#                 f"best_length: {results[(alpha, beta)]['best_length']:.3f}"
#             )
#     return results

# alphas = [3.0, 4.0, 5.0]
# betas  = [5]
# repeats = 2
# iterations = 200
# # grid_search_results = run_grid_search(alphas, betas, n_ants, iterations, Q, evaporation, map)
# # print("Grid Search Results:")
# # for params, metrics in grid_search_results.items():
# #     print(
# #         f"Parameters (alpha={params[0]}, beta={params[1]}): "
# #         f"mean_norm={metrics['mean_norm']:.4f}, best_length={metrics['best_length']:.3f}"
# #     )

# # best_params = min(grid_search_results, key=lambda p: grid_search_results[p]["mean_norm"])
# # print(
# #     f"Best: alpha={best_params[0]}, beta={best_params[1]}, "
# #     f"mean_norm={grid_search_results[best_params]['mean_norm']:.4f}"
# #)


# # print(data["best_paths"])
# def find_srcdest(filename):
#     data = np.load(filename, allow_pickle=True)
#     best_paths = data["best_paths"]
#     fitness_hist = data["fitness_history"]
#     flat_fitness = fitness_hist[:, :, 0]

#     last_gen_idx = -1
#     scores_last_gen = flat_fitness[last_gen_idx]
#     best_colony_idx = np.argmin(scores_last_gen)
#     winning_path = best_paths[last_gen_idx][best_colony_idx]

#     start_node_id = winning_path[0]
#     end_node_id = winning_path[-1]
    
#     return start_node_id, end_node_id

# def run_coarse_to_fine(n_ants, iterations, Q, evaporation, src_dest_nodes ,repeats=5):
    
#     map = osmap.Map()
    
#     coarse_alphas = [0.1, 1, 2, 3, 4, 5]
#     coarse_betas =  [1, 2, 3, 4, 5, 6]
    
#     map.src = src_dest_nodes[0]
#     map.dest = src_dest_nodes[1]
    
#     map.build()
    
#     results_1 = run_grid_search(
#         coarse_alphas, coarse_betas, 
#         n_ants, iterations, Q, evaporation, map, 
#         repeats=repeats
#         )   
 