import argparse
import base64
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import matplotlib
try:
    from IPython import get_ipython
    TASK3_IN_NOTEBOOK = getattr(get_ipython(), "kernel", None) is not None
except ImportError:
    TASK3_IN_NOTEBOOK = False
matplotlib.use("module://matplotlib_inline.backend_inline" if TASK3_IN_NOTEBOOK else "Agg")
import matplotlib.pyplot as plt
if TASK3_IN_NOTEBOOK:
    from IPython.display import Markdown, Image, display

TASK3_MODE = os.environ.get("TASK3_MODE", "run").strip().lower()
TASK3_OUTPUT_DIR = Path(os.environ.get("TASK3_OUTPUT_DIR", "task3_results/baseline-v1"))
if not TASK3_IN_NOTEBOOK:
    task3_parser = argparse.ArgumentParser(description="Run MW7 experiments in Python, then export results for Notebook assembly.")
    task3_parser.add_argument("--mode", choices=("run", "load", "check"), default=TASK3_MODE)
    task3_parser.add_argument("--output-dir", type=Path, default=TASK3_OUTPUT_DIR)
    task3_args = task3_parser.parse_args()
    TASK3_MODE, TASK3_OUTPUT_DIR = task3_args.mode, task3_args.output_dir
if TASK3_MODE not in {"run", "load", "check"}:
    raise ValueError("TASK3_MODE must be run, load, or check")
TASK3_SOURCE_PATH = Path(__file__).resolve() if not TASK3_IN_NOTEBOOK else Path("task3/task3_mw7.py")
TASK3_SOURCE_SHA256 = (hashlib.sha256(TASK3_SOURCE_PATH.read_bytes()).hexdigest()
                      if TASK3_SOURCE_PATH.exists() else None)
TASK3_CAPTURED_OUTPUTS = {}
TASK3_ALGORITHM_VERSION = "task3-cnsga2-v1"
TASK3_BASELINE = dict(pop_size=100, max_gen=500, pc=0.8, pm=0.05,
                      eta_c=20.0, eta_m=20.0, cv_tol=1e-12)
TASK3_SEEDS = (0, 1, 2, 3, 4)
TASK3_SMOKE = dict(TASK3_BASELINE, pop_size=20, max_gen=20)
TASK3_SMOKE_SEED = 2026

def task3_emit_text(tag, text):
    print(text, flush=True)
    TASK3_CAPTURED_OUTPUTS.setdefault(tag, []).append(
        dict(output_type="stream", name="stdout", text=text + "\n"))


def task3_emit_markdown(tag, text):
    if TASK3_IN_NOTEBOOK:
        display(Markdown(text))
    else:
        print(text, flush=True)
    TASK3_CAPTURED_OUTPUTS.setdefault(tag, []).append(
        dict(output_type="display_data", metadata={},
             data={"text/markdown": text, "text/plain": text}))


def task3_emit_figure(tag, path):
    png = path.read_bytes()
    if TASK3_IN_NOTEBOOK:
        display(Image(data=png))
    else:
        print("Saved figure:", path, flush=True)
    TASK3_CAPTURED_OUTPUTS.setdefault(tag, []).append(
        dict(output_type="display_data", metadata={},
             data={"image/png": base64.b64encode(png).decode("ascii"),
                   "text/plain": f"Generated figure: {path.name}"}))


task3_emit_text("task3-config", f"Mode: {TASK3_MODE} | Results: {TASK3_OUTPUT_DIR.resolve()}")
task3_emit_text("task3-config", f"Python: {platform.python_version()} NumPy: {np.__version__} Matplotlib: {matplotlib.__version__}")

# %% [markdown]
# ## 1. Supplied MW7 — unchanged
#
# MW7 has 15 variables in [0,1], two minimization objectives and two constraints.
# Original constraint expressions are feasible when nonnegative. The supplied
# evaluator already converts them to violation magnitudes: it returns F[N,2] and
# nonnegative C[N,2]. Calculate CV=C.sum(axis=1), without negating C again.
# Numerically, CV <= 1e-12 is treated as feasible; strict CV == 0 counts are also
# reported. This tolerance is not a penalty coefficient. Violations are summed
# without rescaling, following the supplied example.

# %% tags=["task3-mw7"]
class MW7():

    def __init__(self):
        self.name = 'MW7'
        self.n_obj = 2  # number of objectives
        self.n_var = 15  # number of decision variables
        self.n_con = 2  # number of constraints

        self.lower = np.zeros(self.n_var)
        self.upper = np.ones(self.n_var)

    def evaluate(self, x):
        pop_size = len(x)
        f = np.zeros((pop_size, self.n_obj))
        c = np.zeros((pop_size, self.n_con))

        g3 = 1 + 2.0 * np.sum((x[:, self.n_obj - 1:] +
                               (x[:, self.n_obj - 2:-1] - 0.5)**2 - 1.0)**2,
                              axis=1)

        f[:, 0] = g3 * x[:, 0]
        f[:, 1] = g3 * (1 - (f[:, 0] / g3)**2)**0.5

        with np.errstate(divide='ignore'):
            l = np.arctan(f[:, 1] / f[:, 0])

        c[:, 0] = (1.2 + 0.4 * (np.sin(4 * l))**16)**2 - f[:, 0]**2 - f[:, 1]**2
        c[:, 1] = f[:, 0]**2 + f[:, 1]**2 - (1.15 - 0.2 * (np.sin(4 * l))**8)**2
        c = np.where(c < 0, -c, 0)  # the degree of constraint violation of x at each constraint

        return f, c

# %% [markdown]
# ## 2. Constrained NSGA-II
#
# The parameter-less constraint rules are: (1) feasible beats infeasible;
# (2) between infeasible candidates, lower total CV wins; (3) between feasible
# candidates, ordinary Pareto dominance applies. Equal-CV infeasible candidates
# do not constraint-dominate each other. Within a front, normalized objective
# crowding breaks ties; exact rank/distance ties are randomized with the run seed.
# Both mating selection and environmental selection use these same rules.

# %% tags=["task3-algorithm"]
class CNSGA2:
    """Constraint-domination NSGA-II for minimization."""

    def __init__(self, pop_size=100, pc=0.8, pm=0.05,
                 eta_c=20.0, eta_m=20.0, cv_tol=1e-12, seed=0):
        if not isinstance(pop_size, (int, np.integer)) or pop_size < 2 or pop_size % 2:
            raise ValueError("pop_size must be an even integer >= 2")
        if not (0 <= pc <= 1 and 0 <= pm <= 1):
            raise ValueError("pc and pm must be in [0,1]")
        if not (np.isfinite(eta_c) and np.isfinite(eta_m) and eta_c > 0 and eta_m > 0):
            raise ValueError("distribution indices must be finite and positive")
        if not np.isfinite(cv_tol) or cv_tol < 0:
            raise ValueError("cv_tol must be finite and nonnegative")
        self.pop_size = int(pop_size)
        self.pc, self.pm = float(pc), float(pm)
        self.eta_c, self.eta_m = float(eta_c), float(eta_m)
        self.cv_tol, self.seed = float(cv_tol), int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.n_evals = 0

    def initialize(self, prob):
        lower, upper = np.asarray(prob.lower), np.asarray(prob.upper)
        if (lower.shape != (prob.n_var,) or upper.shape != lower.shape
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or not np.all(upper > lower)):
            raise ValueError("Expected finite bounds with upper > lower")
        return self.rng.uniform(lower, upper, (self.pop_size, prob.n_var))

    @staticmethod
    def _pareto_dominance(f):
        # Entry (i,j): candidate i dominates candidate j.
        return (np.all(f[:, None, :] <= f[None, :, :], axis=2)
                & np.any(f[:, None, :] < f[None, :, :], axis=2))

    def _constraint_dominance(self, f, cv):
        feasible = cv <= self.cv_tol
        both_feasible = feasible[:, None] & feasible[None, :]
        both_infeasible = ~feasible[:, None] & ~feasible[None, :]
        return ((both_feasible & self._pareto_dominance(f))
                | (feasible[:, None] & ~feasible[None, :])
                | (both_infeasible & (cv[:, None] < cv[None, :])))

    @staticmethod
    def _sort_fronts(dominates):
        rank = np.zeros(len(dominates), dtype=int)
        remaining = dominates.sum(axis=0).astype(int)
        fronts, level = [], 1
        while np.any(rank == 0):
            front = np.flatnonzero((rank == 0) & (remaining == 0))
            if front.size == 0:
                raise RuntimeError("Dominance relation contains a cycle")
            fronts.append(front)
            rank[front] = level
            remaining -= dominates[front].sum(axis=0)
            level += 1
        return rank, fronts

    @staticmethod
    def _crowding_distance(f, fronts):
        distance = np.zeros(len(f), dtype=float)
        for front in fronts:
            if len(front) <= 2:
                distance[front] = np.inf
                continue
            for objective in range(f.shape[1]):
                order = front[np.argsort(f[front, objective], kind="stable")]
                span = f[order[-1], objective] - f[order[0], objective]
                if span == 0:
                    continue
                distance[order[0]] = distance[order[-1]] = np.inf
                distance[order[1:-1]] += (f[order[2:], objective] - f[order[:-2], objective]) / span
        return distance

    def fitness_assignment(self, f, cv):
        f, cv = np.asarray(f, dtype=float), np.asarray(cv, dtype=float)
        if (f.ndim != 2 or len(f) == 0 or f.shape[1] == 0
                or cv.shape != (len(f),) or not np.isfinite(f).all()
                or not np.isfinite(cv).all() or np.any(cv < 0)):
            raise ValueError("Expected finite F[N,M] and nonnegative CV[N]")
        rank, fronts = self._sort_fronts(self._constraint_dominance(f, cv))
        return rank, self._crowding_distance(f, fronts)

    def tournament_selection(self, population_x, rank, crowding_dist):
        mating_pool = np.empty_like(population_x)
        for i in range(self.pop_size):
            a, b = self.rng.choice(len(population_x), 2, replace=False)
            if rank[a] != rank[b]:
                winner = a if rank[a] < rank[b] else b
            elif crowding_dist[a] != crowding_dist[b]:
                winner = a if crowding_dist[a] > crowding_dist[b] else b
            else:
                winner = int(self.rng.choice([a, b]))
            mating_pool[i] = population_x[winner]
        return mating_pool

    def crossover(self, parents, prob):
        children = parents.copy()
        for i in range(0, self.pop_size, 2):
            if self.rng.random() <= self.pc:
                p1, p2 = parents[i], parents[i + 1]
                u = self.rng.random(prob.n_var)
                beta = np.where(u <= 0.5,
                                (2 * u) ** (1 / (self.eta_c + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
                children[i] = np.clip(
                    0.5 * ((1 + beta) * p1 + (1 - beta) * p2), prob.lower, prob.upper)
                children[i + 1] = np.clip(
                    0.5 * ((1 - beta) * p1 + (1 + beta) * p2), prob.lower, prob.upper)
        return children

    def mutation(self, offspring, prob):
        children = offspring.copy()
        for i in range(self.pop_size):
            for j in range(prob.n_var):
                if self.rng.random() <= self.pm:
                    value = children[i, j]
                    lower, upper = prob.lower[j], prob.upper[j]
                    delta1 = (value - lower) / (upper - lower)
                    delta2 = (upper - value) / (upper - lower)
                    u, power = self.rng.random(), 1 / (self.eta_m + 1)
                    if u < 0.5:
                        v = 2 * u + (1 - 2 * u) * (1 - delta1) ** (self.eta_m + 1)
                        delta = v ** power - 1
                    else:
                        v = 2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (self.eta_m + 1)
                        delta = 1 - v ** power
                    children[i, j] = np.clip(value + delta * (upper - lower), lower, upper)
        return children

    def _evaluate(self, prob, x):
        if (x.shape != (self.pop_size, prob.n_var) or not np.isfinite(x).all()
                or np.any(x < prob.lower) or np.any(x > prob.upper)):
            raise ValueError("Invalid decision vectors or variable bounds")
        f, c = prob.evaluate(x)
        f, c = np.asarray(f, dtype=float), np.asarray(c, dtype=float)
        if (f.shape != (len(x), prob.n_obj) or c.shape != (len(x), prob.n_con)
                or not np.isfinite(f).all() or not np.isfinite(c).all() or np.any(c < 0)):
            raise ValueError("Invalid objectives or constraint violation magnitudes")
        self.n_evals += len(x)
        return f, c, c.sum(axis=1)

    def environmental_selection(self, x, f, c, offspring_x, offspring_f, offspring_c):
        combined_x = np.vstack((x, offspring_x))
        combined_f = np.vstack((f, offspring_f))
        combined_c = np.vstack((c, offspring_c))
        combined_cv = combined_c.sum(axis=1)
        rank, distance = self.fitness_assignment(combined_f, combined_cv)
        tie_break = self.rng.random(len(combined_x))
        selected = np.lexsort((tie_break, -distance, rank))[:self.pop_size]
        return (combined_x[selected], combined_f[selected],
                combined_c[selected], combined_cv[selected])

    def feasible_nd_mask(self, f, cv):
        indices = np.flatnonzero(cv <= self.cv_tol)
        selected = np.zeros(len(f), dtype=bool)
        if len(indices):
            dominated = np.any(self._pareto_dominance(f[indices]), axis=0)
            selected[indices[~dominated]] = True
        return selected

    def _record(self, history, generation, f, cv):
        history.append(dict(generation=generation, n_evals=self.n_evals,
                            feasible_ratio=float(np.mean(cv <= self.cv_tol)),
                            strict_feasible_ratio=float(np.mean(cv == 0)),
                            min_cv=float(cv.min()), mean_cv=float(cv.mean()),
                            max_cv=float(cv.max()),
                            feasible_nd_count=int(self.feasible_nd_mask(f, cv).sum())))

    def run(self, prob, max_gen=500, progress_every=100):
        if not isinstance(max_gen, (int, np.integer)) or max_gen < 0:
            raise ValueError("max_gen must be a nonnegative integer")
        self.rng = np.random.default_rng(self.seed)
        self.n_evals = 0
        started = time.perf_counter()
        x = self.initialize(prob)
        f, c, cv = self._evaluate(prob, x)
        history = []
        self._record(history, 0, f, cv)
        for generation in range(1, max_gen + 1):
            rank, distance = self.fitness_assignment(f, cv)
            parents = self.tournament_selection(x, rank, distance)
            offspring_x = self.mutation(self.crossover(parents, prob), prob)
            offspring_f, offspring_c, _ = self._evaluate(prob, offspring_x)
            x, f, c, cv = self.environmental_selection(x, f, c, offspring_x, offspring_f, offspring_c)
            self._record(history, generation, f, cv)
            if progress_every and (generation % progress_every == 0 or generation == max_gen):
                print(f"seed={self.seed}, update={generation}/{max_gen}, "
                      f"feasible={np.mean(cv <= self.cv_tol):.0%}, min_CV={cv.min():.3g}", flush=True)
        return dict(x=x, f=f, c=c, cv=cv, nd_mask=self.feasible_nd_mask(f, cv),
                    history=history, n_evals=self.n_evals,
                    elapsed_seconds=time.perf_counter() - started)

# %% [markdown]
# ## 3. Correctness checks (no evolution)
#
# These exercise known dominance cases, equal CV, all-feasible/all-infeasible
# populations, crowding, mating, survivor/data alignment, variation bounds and
# the supplied evaluator

# %% tags=["task3-checks"]
def task3_run_checks():
    opt = CNSGA2(pop_size=2, seed=7)
    f = np.array([[10., 10.], [-10., -10.]])
    # Feasibility wins despite worse objectives.
    rank, _ = opt.fitness_assignment(f, np.array([0., 1.]))
    np.testing.assert_array_equal(rank, [1, 2])
    # All infeasible: smaller CV wins; equal CV implies no dominance.
    rank, _ = opt.fitness_assignment(f, np.array([0.1, 1.]))
    np.testing.assert_array_equal(rank, [1, 2])
    rank, _ = opt.fitness_assignment(f, np.array([1., 1.]))
    np.testing.assert_array_equal(rank, [1, 1])
    # All feasible: usual Pareto ranking, boundary and interior crowding.
    f = np.array([[0., 2.], [1., 1.], [2., 0.], [2., 2.]])
    rank, distance = opt.fitness_assignment(f, np.zeros(4))
    np.testing.assert_array_equal(rank, [1, 1, 1, 2])
    assert np.isinf(distance[0]) and np.isinf(distance[2])
    np.testing.assert_allclose(distance[1], 2.)
    rank, distance = opt.fitness_assignment(np.ones((4, 2)), np.zeros(4))
    np.testing.assert_array_equal(rank, [1, 1, 1, 1])
    assert not np.isnan(distance).any()
    rank, _ = opt.fitness_assignment(np.array([[10., 10.], [0., 0.]]),
                                     np.array([opt.cv_tol, 2 * opt.cv_tol]))
    np.testing.assert_array_equal(rank, [1, 2])
    chosen = opt.tournament_selection(np.array([[1.], [2.]]),
                                      np.array([1, 2]), np.array([0., np.inf]))
    np.testing.assert_array_equal(chosen, [[1.], [1.]])
    # Feasible offspring beat infeasible parents; X/F/C stay aligned.
    x, f, c, cv = opt.environmental_selection(
        np.array([[0.], [1.]]), np.array([[-10., -10.], [-9., -9.]]),
        np.array([[1., 0.], [2., 0.]]),
        np.array([[2.], [3.]]), np.array([[10., 11.], [11., 10.]]), np.zeros((2, 2)))
    np.testing.assert_array_equal(np.sort(x[:, 0]), [2., 3.])
    assert np.all(cv == 0) and np.all(c == 0)
    for row_x, row_f in zip(x, f):
        np.testing.assert_array_equal(row_f, [10., 11.] if row_x[0] == 2 else [11., 10.])
    np.testing.assert_array_equal(opt.feasible_nd_mask(f, cv), [True, True])
    np.testing.assert_array_equal(opt.feasible_nd_mask(f, np.ones(2)), [False, False])
    # Bound-safe variation and reproducible initialization, without evolving.
    prob = MW7()
    a = CNSGA2(pop_size=20, pc=1., pm=1., seed=123)
    b = CNSGA2(pop_size=20, pc=1., pm=1., seed=123)
    initial = a.initialize(prob)
    np.testing.assert_array_equal(initial, b.initialize(prob))
    children = a.mutation(a.crossover(initial, prob), prob)
    assert children.shape == (20, 15) and np.isfinite(children).all()
    assert np.all(children >= prob.lower) and np.all(children <= prob.upper)
    f, c = prob.evaluate(np.vstack((np.zeros(15), np.ones(15), np.full(15, 0.5))))
    assert f.shape == (3, 2) and c.shape == (3, 2)
    assert np.isfinite(f).all() and np.isfinite(c).all() and np.all(c >= 0)
    for invalid in (1, 3):
        try:
            CNSGA2(pop_size=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid population size accepted")
    task3_emit_text("task3-checks", "Deterministic checks passed. No evolution was run in these checks.")

task3_run_checks()

# %% [markdown]
# ## 4. Persistence and validation
# %% tags=["task3-persistence"]
def task3_optimizer(config, seed):
    return CNSGA2(seed=seed, **{k: v for k, v in config.items() if k != "max_gen"})


def task3_metadata(config, seed):
    return dict(schema_version=1, algorithm_version=TASK3_ALGORITHM_VERSION,
                problem="MW7", parameters=dict(config), seed=int(seed),
                python=platform.python_version(), numpy=np.__version__,
                matplotlib=matplotlib.__version__,
                source_python_sha256=TASK3_SOURCE_SHA256)


def task3_validate_result(result, config, seed):
    meta = result["metadata"]
    for key, expected in (("schema_version", 1), ("algorithm_version", TASK3_ALGORITHM_VERSION),
                          ("problem", "MW7"), ("parameters", config), ("seed", int(seed))):
        if meta.get(key) != expected:
            raise ValueError(f"Saved {key} differs; use a new output directory")
    n = config["pop_size"]
    for name, shape in (("x", (n, 15)), ("f", (n, 2)), ("c", (n, 2)), ("cv", (n,))):
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise ValueError(f"Invalid saved {name}")
    if np.any(result["x"] < 0) or np.any(result["x"] > 1) or np.any(result["c"] < 0):
        raise ValueError("Invalid bounds or CV convention")
    f, c = MW7().evaluate(result["x"])
    np.testing.assert_allclose(result["f"], f, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["c"], c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["cv"], result["c"].sum(axis=1), rtol=1e-12, atol=1e-12)
    nd = task3_optimizer(config, seed).feasible_nd_mask(result["f"], result["cv"])
    np.testing.assert_array_equal(result["nd_mask"], nd)
    if result["n_evals"] != n * (config["max_gen"] + 1):
        raise ValueError("Unexpected search evaluation budget")
    history = result["history"]
    if len(history) != config["max_gen"] + 1:
        raise ValueError("Incomplete saved history")
    for generation, row in enumerate(history):
        if row["generation"] != generation or row["n_evals"] != n * (generation + 1):
            raise ValueError("Incorrect generation/evaluation count")
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError("Non-finite history")
        if not (0 <= row["strict_feasible_ratio"] <= row["feasible_ratio"] <= 1
                and 0 <= row["feasible_nd_count"] <= n
                and row["min_cv"] >= 0
                and row["min_cv"] <= row["mean_cv"] + 1e-12 * max(1., row["max_cv"])
                and row["mean_cv"] <= row["max_cv"] + 1e-12 * max(1., row["max_cv"])):
            raise ValueError("Invalid history statistics")
    np.testing.assert_allclose(history[-1]["feasible_ratio"], np.mean(result["cv"] <= config["cv_tol"]))
    np.testing.assert_allclose(history[-1]["strict_feasible_ratio"], np.mean(result["cv"] == 0))
    for key, expected in (("min_cv", result["cv"].min()), ("mean_cv", result["cv"].mean()),
                          ("max_cv", result["cv"].max())):
        np.testing.assert_allclose(history[-1][key], expected)
    assert history[-1]["feasible_nd_count"] == int(nd.sum())
    if not np.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"] < 0:
        raise ValueError("Invalid elapsed time")


def task3_save_result(path, result):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, x=result["x"], f=result["f"], c=result["c"],
                            cv=result["cv"], nd_mask=result["nd_mask"],
                            history_json=np.array(json.dumps(result["history"])),
                            metadata_json=np.array(json.dumps(result["metadata"], sort_keys=True)),
                            n_evals=np.array(result["n_evals"]),
                            elapsed_seconds=np.array(result["elapsed_seconds"]))
    temporary.replace(path)


def task3_load_result(path):
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in ("x", "f", "c", "cv", "nd_mask")}
        result.update(history=json.loads(str(data["history_json"].item())),
                      metadata=json.loads(str(data["metadata_json"].item())),
                      n_evals=int(data["n_evals"].item()),
                      elapsed_seconds=float(data["elapsed_seconds"].item()))
    return result


def task3_export_csv(path, result):
    feasible = result["cv"] <= result["metadata"]["parameters"]["cv_tol"]
    header = [f"x{i + 1}" for i in range(15)] + ["f1", "f2", "c1", "c2", "CV", "feasible", "feasible_nondominated"]
    data = np.column_stack((result["x"], result["f"], result["c"], result["cv"],
                            feasible.astype(int), result["nd_mask"].astype(int)))
    np.savetxt(path.with_name(path.stem + "_population.csv"), data, delimiter=",",
               header=",".join(header), comments="", fmt="%.17g")
    with path.with_name(path.stem + "_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["history"][0]))
        writer.writeheader()
        writer.writerows(result["history"])


def task3_get_result(name, config, seed, mode):
    if mode not in {"run", "load"}:
        raise ValueError("Experiment mode must be run or load")
    path = TASK3_OUTPUT_DIR / (name + ".npz")
    if path.exists():
        print("Loading completed run:", path.name, flush=True)
        result = task3_load_result(path)
    elif mode == "load":
        raise FileNotFoundError(f"Missing {path}; load mode never starts optimization. Run experiments yourself with TASK3_MODE=run.")
    else:
        TASK3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Starting {name}: {config}", flush=True)
        result = task3_optimizer(config, seed).run(MW7(), max_gen=config["max_gen"], progress_every=100)
        result["metadata"] = task3_metadata(config, seed)
        task3_validate_result(result, config, seed)
        task3_save_result(path, result)
        print(f"Saved {path.name} ({result['elapsed_seconds']:.1f} s)", flush=True)
    task3_validate_result(result, config, seed)
    task3_export_csv(path, result)
    tag = "task3-experiments" if name.startswith("smoke_") else f"task3-seed-{seed}"
    task3_emit_text(tag, f"{name}: evaluations={result['n_evals']}, "
                    f"feasible={np.sum(result['cv'] <= config['cv_tol'])}/{config['pop_size']}, "
                    f"feasible nondominated={result['nd_mask'].sum()}, "
                    f"search time={result['elapsed_seconds']:.1f} s")
    return result

# %% [markdown]
# ## 5. MW7 experiments
# %% tags=["task3-experiments"]
task3_results = []
if TASK3_MODE == "check":
    print("Check-only: smoke and baseline optimizations are skipped.")
elif TASK3_MODE == "run":
    task3_smoke_result = task3_get_result("smoke_seed_2026", TASK3_SMOKE, TASK3_SMOKE_SEED, "run")
    print("Smoke validated. Starting/loading the five baseline runs.")
else:
    print("Load-only: no optimization will run; loading five saved baselines.")

# %% tags=["task3-experiments", "task3-seed-0"]
if TASK3_MODE != "check":
    task3_results.append(task3_get_result("seed_0", TASK3_BASELINE, 0, TASK3_MODE))

# %% tags=["task3-experiments", "task3-seed-1"]
if TASK3_MODE != "check":
    task3_results.append(task3_get_result("seed_1", TASK3_BASELINE, 1, TASK3_MODE))

# %% tags=["task3-experiments", "task3-seed-2"]
if TASK3_MODE != "check":
    task3_results.append(task3_get_result("seed_2", TASK3_BASELINE, 2, TASK3_MODE))

# %% tags=["task3-experiments", "task3-seed-3"]
if TASK3_MODE != "check":
    task3_results.append(task3_get_result("seed_3", TASK3_BASELINE, 3, TASK3_MODE))

# %% tags=["task3-experiments", "task3-seed-4"]
if TASK3_MODE != "check":
    task3_results.append(task3_get_result("seed_4", TASK3_BASELINE, 4, TASK3_MODE))

# %% [markdown]
# ## 6. Tables and figures from numerical results

# %% tags=["task3-analysis", "task3-summary"]
def task3_summary_row(result):
    tol = result["metadata"]["parameters"]["cv_tol"]
    return dict(seed=result["metadata"]["seed"],
                feasible_count=int(np.sum(result["cv"] <= tol)),
                strict_feasible_count=int(np.sum(result["cv"] == 0)),
                feasible_nd_count=int(result["nd_mask"].sum()),
                min_cv=float(result["cv"].min()), mean_cv=float(result["cv"].mean()),
                max_cv=float(result["cv"].max()), n_evals=result["n_evals"],
                elapsed_seconds=result["elapsed_seconds"])


task3_summary = [task3_summary_row(result) for result in task3_results]
if task3_summary:
    with (TASK3_OUTPUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task3_summary[0]))
        writer.writeheader()
        writer.writerows(task3_summary)
    task3_table = "| Seed | Feasible | Strict feasible | Feasible ND | Mean CV | Max CV | Evaluations | Seconds |\n"
    task3_table += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for row in task3_summary:
        task3_table += (f"| {row['seed']} | {row['feasible_count']} | {row['strict_feasible_count']} "
                       f"| {row['feasible_nd_count']} | {row['mean_cv']:.3g} | {row['max_cv']:.3g} "
                       f"| {row['n_evals']} | {row['elapsed_seconds']:.1f} |\n")
    task3_emit_markdown("task3-summary", task3_table)
else:
    print("No experimental results in check-only mode.")

# %% tags=["task3-plots"]
def task3_scatter(ax, result):
    f, cv = result["f"], result["cv"]
    feasible = cv <= result["metadata"]["parameters"]["cv_tol"]
    nd = result["nd_mask"]
    for mask, color, marker, label, size in (
            (~feasible, "#c44e52", "x", "Infeasible", 26),
            (feasible & ~nd, "#999999", "o", "Feasible dominated", 18),
            (nd, "#2878b5", "o", "Feasible nondominated", 22)):
        if mask.any():
            ax.scatter(f[mask, 0], f[mask, 1], color=color, marker=marker, s=size, label=label)
    ax.set(xlabel="$f_1$", ylabel="$f_2$",
           title=f"Seed {result['metadata']['seed']} | feasible {feasible.sum()}/{len(f)} | ND {nd.sum()}")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7)


if task3_results:
    task3_figure_dir = TASK3_OUTPUT_DIR / "figures"
    task3_figure_dir.mkdir(exist_ok=True)
    task3_all_f = np.vstack([r["f"] for r in task3_results])
    task3_span = np.maximum(np.ptp(task3_all_f, axis=0), 0.1)
    task3_lower = task3_all_f.min(axis=0) - 0.05 * task3_span
    task3_upper = task3_all_f.max(axis=0) + 0.05 * task3_span

    task3_fig, task3_ax = plt.subplots(figsize=(6.5, 5))
    task3_scatter(task3_ax, task3_results[0])
    task3_ax.set_xlim(task3_lower[0], task3_upper[0])
    task3_ax.set_ylim(task3_lower[1], task3_upper[1])
    task3_fig.tight_layout()
    task3_fig.savefig(task3_figure_dir / "baseline_seed_0.png", dpi=180)
    task3_emit_figure("task3-plots", task3_figure_dir / "baseline_seed_0.png")
    plt.close(task3_fig)

    task3_fig, task3_axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for task3_ax, task3_result in zip(task3_axes.flat, task3_results):
        task3_scatter(task3_ax, task3_result)
        task3_ax.set_xlim(task3_lower[0], task3_upper[0])
        task3_ax.set_ylim(task3_lower[1], task3_upper[1])
    task3_axes.flat[-1].axis("off")
    task3_fig.suptitle("MW7 — all five preselected seeds (shared objective scales)")
    task3_fig.tight_layout()
    task3_fig.savefig(task3_figure_dir / "all_seeds.png", dpi=180)
    task3_emit_figure("task3-plots", task3_figure_dir / "all_seeds.png")
    plt.close(task3_fig)

    task3_fig, task3_axes = plt.subplots(1, 3, figsize=(15, 4))
    for task3_result in task3_results:
        h = task3_result["history"]
        generations = [row["generation"] for row in h]
        label = f"seed {task3_result['metadata']['seed']}"
        task3_axes[0].plot(generations, [row["feasible_ratio"] for row in h], label=label)
        task3_axes[1].plot(generations, [row["mean_cv"] for row in h], label=label)
        task3_axes[2].plot(generations, [row["feasible_nd_count"] for row in h], label=label)
    task3_axes[0].set(ylabel="Feasible proportion", ylim=(-0.03, 1.03))
    task3_axes[1].set_ylabel("Mean CV")
    task3_axes[1].set_yscale("symlog", linthresh=1e-12)
    task3_axes[2].set_ylabel("Feasible nondominated count")
    for task3_ax in task3_axes:
        task3_ax.set_xlabel("Evolutionary update (0 = initialization)")
        task3_ax.grid(alpha=0.25)
        task3_ax.legend(fontsize=8)
    task3_fig.tight_layout()
    task3_fig.savefig(task3_figure_dir / "search_history.png", dpi=180)
    task3_emit_figure("task3-plots", task3_figure_dir / "search_history.png")
    plt.close(task3_fig)

if task3_summary:
    task3_feasible_counts = np.array([row["feasible_count"] for row in task3_summary])
    task3_nd_counts = np.array([row["feasible_nd_count"] for row in task3_summary])
    task3_observations = (
        "# Task 3 — measured observations\n\n"
        f"Parameters: {json.dumps(TASK3_BASELINE, sort_keys=True)}. Seeds: {list(TASK3_SEEDS)}.\n\n"
        + task3_table + "\n"
        + f"Feasible count: mean {task3_feasible_counts.mean():.1f}, "
          f"sample SD {task3_feasible_counts.std(ddof=1):.1f}, "
          f"range {task3_feasible_counts.min()}–{task3_feasible_counts.max()}.\n\n"
        + f"Feasible nondominated count: mean {task3_nd_counts.mean():.1f}, "
          f"sample SD {task3_nd_counts.std(ddof=1):.1f}, "
          f"range {task3_nd_counts.min()}–{task3_nd_counts.max()}.\n\n"
        + "These counts describe feasibility and within-population nondominance, "
          "not convergence to the true PF or full segment coverage. All preselected "
          "seeds are shown. Interpret spatial gaps qualitatively until a validated "
          "numerical reference front is available.\n"
    )
    (TASK3_OUTPUT_DIR / "observations.md").write_text(task3_observations, encoding="utf-8")
    (TASK3_OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(dict(algorithm_version=TASK3_ALGORITHM_VERSION, baseline=TASK3_BASELINE,
                        seeds=list(TASK3_SEEDS), mode=TASK3_MODE,
                        runs=[r["metadata"] for r in task3_results]), indent=2), encoding="utf-8")
    task3_emit_markdown("task3-observations", task3_observations)
    print("Artifacts ready for review:", TASK3_OUTPUT_DIR.resolve())

# %% tags=["task3-export"]
if task3_results:
    if [r["metadata"]["seed"] for r in task3_results] != list(TASK3_SEEDS):
        raise ValueError("Cannot export an incomplete or reordered baseline experiment")
    task3_artifact_names = [f"seed_{seed}.npz" for seed in TASK3_SEEDS]
    task3_artifact_names += [f"seed_{seed}_{suffix}.csv" for seed in TASK3_SEEDS
                            for suffix in ("population", "history")]
    task3_artifact_names += ["summary.csv", "observations.md", "experiment_manifest.json"]
    task3_artifact_names += ["figures/" + name for name in
                            ("baseline_seed_0.png", "all_seeds.png", "search_history.png")]
    task3_bundle = dict(schema_version=1, producer="task3/task3_mw7.py",
                        source_sha256=TASK3_SOURCE_SHA256,
                        algorithm_version=TASK3_ALGORITHM_VERSION,
                        mode=TASK3_MODE, baseline=TASK3_BASELINE, seeds=list(TASK3_SEEDS),
                        outputs_by_tag=TASK3_CAPTURED_OUTPUTS,
                        artifacts={name: hashlib.sha256((TASK3_OUTPUT_DIR / name).read_bytes()).hexdigest()
                                   for name in task3_artifact_names})
    task3_bundle_tmp = TASK3_OUTPUT_DIR / "notebook_outputs.tmp"
    task3_bundle_tmp.write_text(json.dumps(task3_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    task3_bundle_tmp.replace(TASK3_OUTPUT_DIR / "notebook_outputs.json")
    print("Python run complete. Assemble the Notebook without rerunning experiments:")
    print(f"python scripts/build_task3_notebook.py --results {TASK3_OUTPUT_DIR}")
