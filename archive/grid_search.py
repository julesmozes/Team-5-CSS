import minimal.ants as ants
import minimal.osmap as osmap
import numpy as np
import networkx as nx
from scipy.stats import mannwhitneyu
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os

_GS = None

def _init_worker(gs_kwargs, backup_path, verbose=False):
    global _GS
    _GS = Grid_search(**gs_kwargs)
    _GS.verbose = verbose
    _GS.build_map(filename=backup_path)

def _eval_pairs(pairs):
    # evaluates a chunk of pairs in one worker
    out = []
    for a, b in pairs:
        if getattr(_GS, "verbose", False):
            print(f"testing a={a:.3f}, b={b:.3f}", flush=True)
        out.append(((a, b), _GS.eval_one(a, b)))
    return out


class Grid_search():
    def __init__(self, n_ants, iterations, Q, evaporation, init_params, repeats=3, base_seed=67):
        self.n_ants = n_ants
        self.iterations = iterations
        self.Q = Q
        self.evaporation = evaporation
        self.alphas = init_params[0]
        self.betas = init_params[1]
        self.repeats = repeats
        self.base_seed = base_seed
        self.evolution_data = None
        
        
        
    def run_manual_grid_search(self, backup_path, max_workers=6, chunk_size=25, verbose=False):
        pairs = list(product(self.alphas, self.betas))

        # chunk the work to reduce overhead
        chunks = [pairs[i:i+chunk_size] for i in range(0, len(pairs), chunk_size)]

        gs_kwargs = dict(
            n_ants=self.n_ants,
            iterations=self.iterations,
            Q=self.Q,
            evaporation=self.evaporation,
            init_params=([1.0], [1.0]),  # dummy
            repeats=self.repeats,
            base_seed=self.base_seed,
        )

        results = {}
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(gs_kwargs, backup_path, verbose),
        ) as ex:
            for chunk_out in ex.map(_eval_pairs, chunks):
                for (a, b), metrics in chunk_out:
                    results[(a, b)] = metrics

        return results
    
    
    
    def eval_one(self, a, b):
        solved_iterations = []
        best_lengths = []

        for r in range(self.repeats):
            np.random.seed(self.base_seed + r)
            self.map.resetPheromones()

            iters_solved, best_length = self.eval_one_single(a, b)
            solved_iterations.append(iters_solved)
            best_lengths.append(best_length)

        return {
            "fitness": float(np.mean(solved_iterations)),
            "best_length": float(np.mean(best_lengths)),
        }
        
        
        
    def eval_one_single(self, a, b):
        target_dist = self.min_dist + 50.0
        solved = False
        iters_solved = self.iterations
        best_length = float("inf")
        
        for it in range(self.iterations):
                solutions = []

                for i in range(self.n_ants):
                    path, length, steps = ants.build_path_numba(
                        self.map.getNumbaData(), a, b, max_steps=1000
                    )
                    if steps:
                        solutions.append((path, length, steps))

                if not solutions:
                    continue

                ants.update_pheromones_numba(
                    self.map.pheromone, solutions, self.Q, self.evaporation
                )

                current_best = min(length for _, length, _ in solutions)
                if current_best < best_length:
                    best_length = current_best

                if not solved and current_best <= target_dist:
                    iters_solved = it
                    solved = True
                    break
                
        return iters_solved, best_length
    
    
    
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
        self.map.build("Amsterdam, Netherlands", add_travel_time=False)
        
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
        return min(results, key=lambda p: (results[p]["fitness"], results[p]["best_length"]))
    
    
    
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
            (best_a, best_b), best_metrics, top = self.run_random_search(n_samples=500, top_k=10)
            old_iters, old_repeats = self.iterations, self.repeats
            self.iterations = 600 
            self.repeats = 5         

            results_coarse = {}
            for _, _, a, b, _m in top:
                results_coarse[(a, b)] = self.eval_one(a, b)

            best_a, best_b = self.pick_best(results_coarse)
            best_metrics = results_coarse[(best_a, best_b)]

            self.iterations, self.repeats = old_iters, old_repeats

        else:
            raise ValueError("mode must be 'manual' or 'random' :((")

        print("\nBEST coarse:", best_a, best_b, best_metrics)

        fine_alphas, fine_betas = self.make_fine_grid(best_a, best_b, fine_step_a, fine_step_b)

        old_alphas, old_betas = self.alphas, self.betas
        self.alphas, self.betas = fine_alphas, fine_betas

        old_iters, old_repeats = self.iterations, self.repeats
        self.iterations = 1000  
        self.repeats = 5        

        results_fine = self.run_manual_grid_search()
        best_a2, best_b2 = self.pick_best(results_fine)

        print("\nBEST fine:", best_a2, best_b2, results_fine[(best_a2, best_b2)])

        self.alphas, self.betas = old_alphas, old_betas
        self.iterations, self.repeats = old_iters, old_repeats

        return results_coarse, results_fine, (best_a2, best_b2)
    
    
    
    def run_random_search(self, n_samples=30, alpha_range=(0.5, 5.0), beta_range=(0.75, 6.0),
                      fast_iters=300, fast_repeats=3, top_k=5):

        old_iters, old_repeats = self.iterations, self.repeats
        self.iterations = fast_iters
        self.repeats = fast_repeats

        rng = np.random.default_rng(self.base_seed)

        top = []  # list of tuples: (fitness, best_length, a, b, metrics)

        for _ in range(n_samples):
            a = float(rng.uniform(*alpha_range))
            b = float(rng.uniform(*beta_range))

            metrics = self.eval_one(a, b)
            key = (metrics["fitness"], metrics["best_length"])

            top.append((key[0], key[1], a, b, metrics))
            
            top.sort(key=lambda t: (t[0], t[1]))
            top = top[:top_k]

            print(f"try a={a:.3f}, b={b:.3f} -> fitness={metrics['fitness']:.2f}")

        self.iterations, self.repeats = old_iters, old_repeats

        best = top[0]
        best_params = (best[2], best[3])
        best_metrics = best[4]

        print("\nTOP candidates:")
        for rank, t in enumerate(top, 1):
            print(f"{rank}) a={t[2]:.3f}, b={t[3]:.3f} -> fitness={t[0]:.2f}, best_len={t[1]:.2f}")

        print("\nBEST random:", best_params, best_metrics)
        return best_params, best_metrics, top
        
        
    
    def run_colony_dna(self, num_evals, num_iterations, num_ants):
        colony = self.evo_data["best_individual"][-2] # laatst gevulde index blijkbaar -1 is leeg, maar hoort geen probleem te zijn lijkt me
        np.random.seed()

        target_dist = self.min_dist + 50.0
        its_to_threshold = []
        for _ in range(num_evals):
            self.map.resetPheromones()
            reached_it = num_iterations  # default = failure

            for it_idx in range(num_iterations):
                solutions = []

                for ant_idx in range(num_ants):
                    path, length, steps = ants.build_path_numba(
                        self.map.getNumbaData(),
                        colony[ant_idx, 0],
                        colony[ant_idx, 1],
                    )
                    if steps:
                        solutions.append((path, length, steps))

                if not solutions:
                    continue

                ants.update_pheromones_numba(
                    self.map.pheromone,
                    solutions,
                    Q=self.Q,
                    evaporation=self.evaporation,
                )

                best_length = min(length for _, length, _ in solutions)
                if best_length <= target_dist:
                    reached_it = it_idx
                    break

            its_to_threshold.append(reached_it)

        self.its_to_threshold = its_to_threshold
        return its_to_threshold
    
    
    
    def sample_grid(self, alpha, beta, num_evals=100):
        np.random.seed()
        samples = []
        for _ in range(num_evals):
            self.map.resetPheromones()
            iters_solved, _best_len = self.eval_one_single(alpha, beta)
            samples.append(iters_solved)
        return samples
        
        
        
    def cliffs_delta(self, x, y):
        n_x = len(x)
        n_y = len(y)
        greater = sum(xi > yi for xi in x for yi in y)
        less = sum(xi < yi for xi in x for yi in y)
        return (greater - less) / (n_x * n_y)


    def compare_evo_vs_grid(self, best_alpha, best_beta, num_evals=30, alternative="greater"):
        """
        tests H1: grid needs more iterations than evo (evo better).
        """

        evo_samples = self.run_colony_dna(
            num_evals=num_evals,
            num_iterations=self.iterations,
            num_ants=self.n_ants,
        )
        grid_samples = self.sample_grid(best_alpha, best_beta, num_evals=num_evals)

        u_stat, p_val = mannwhitneyu(grid_samples, evo_samples, alternative=alternative)

        
        delta = self.cliffs_delta(grid_samples, evo_samples)

        report = {
            "best_alpha": float(best_alpha),
            "best_beta": float(best_beta),
            "n": int(num_evals),
            "alternative": alternative,
            "U": float(u_stat),
            "p": float(p_val),
            "cliffs_delta": float(delta),
            "grid_mean": float(np.mean(grid_samples)),
            "evo_mean": float(np.mean(evo_samples)),
            "grid_median": float(np.median(grid_samples)),
            "evo_median": float(np.median(evo_samples)),
        }
        return report, grid_samples, evo_samples




if __name__ == "__main__":
    alphas = np.arange(0.9, 2.01, 0.01)
    betas  = np.arange(1, 6.01, 0.05)
    init_params = (alphas, betas)

    n_ants = 100
    iterations = 300
    Q = 1
    evaporation = 0.2

    backup = "new_version_colony_evolution/evolution_backup.npz"

    grid_search = Grid_search(n_ants, iterations, Q, evaporation, init_params)
    grid_search.build_map(filename=backup)
    compare_report, grid_samples, evo_samples = grid_search.compare_evo_vs_grid(1.697, 2.773, num_evals=100)
   

    timestamp = np.datetime64("now").astype(str).replace(":", "-")
    out_path = f"manual_grid_{timestamp}.npz"
    compare_out_path = f"compare_evo_vs_grid_{timestamp}.npz"

    # results = grid_search.run_manual_grid_search(backup_path=backup)
    # best_a, best_b = grid_search.pick_best(results)
    # best_metrics = results[(best_a, best_b)]
    # print("\nBEST manual:", best_a, best_b, best_metrics)
    # ranodom compare
    # np.savez(
    #     out_path,
    #     results=np.array(list(results.items()), dtype=object),
    #     best_alpha=float(best_a),
    #     best_beta=float(best_b),
    #     best_metrics=np.array(best_metrics, dtype=object),
    #     alphas=alphas,
    #     betas=betas,
    #     src_node=int(grid_search.map.src_node),
    #     dest_node=int(grid_search.map.dest_node),
    #     min_dist=float(grid_search.min_dist),
    #     backup_path=backup,
    #     n_ants=int(n_ants),
    #     iterations=int(iterations),
    #     Q=float(Q),
    #     evaporation=float(evaporation),
    # )

    # manual compare save
    # np.savez(compare_out_path,
    #     report=np.array(compare_report, dtype=object),
    #     best_alpha=float(compare_report["best_alpha"]),
    #     best_beta=float(compare_report["best_beta"]),
    #     grid_samples=np.array(grid_samples, dtype=int),
    #     evo_samples=np.array(evo_samples, dtype=int),
    #     src_node=int(grid_search.map.src_node),
    # )
    # print(f"Saved results to: {out_path}")
    
    # grid_search.run_colony_dna(num_evals=20, num_iterations=1000, num_ants=100)
    
