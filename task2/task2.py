"""Task 2 crashworthiness experiment using the group's existing classes.

Pass the existing NSGA2 and supplied CrashworthinessDesign classes to run_task2.
"""

import time

import matplotlib.pyplot as plt
import numpy as np


def run_task2(NSGA2, CrashworthinessDesign, *, seed=2026, pop_size=100, max_gen=500):
    """Run, check and plot Task 2; return designs, objectives and runtime."""
    # NSGA-II run
    crash_problem = CrashworthinessDesign()
    CRASH_SEED = seed
    CRASH_POP_SIZE = pop_size
    CRASH_MAX_GEN = max_gen
    np.random.seed(CRASH_SEED)
    crash_optimizer = NSGA2(pop_size=CRASH_POP_SIZE)
    start_time = time.perf_counter()
    crash_x, crash_f = crash_optimizer.run(crash_problem, max_gen=CRASH_MAX_GEN)
    crash_runtime_seconds = time.perf_counter() - start_time
    print(f'Seed={CRASH_SEED}, population={CRASH_POP_SIZE}, max_gen={CRASH_MAX_GEN}')
    print(f'Elapsed time: {crash_runtime_seconds:.1f} s')

    # Result checks
    assert crash_x.shape == (CRASH_POP_SIZE, crash_problem.n_var)
    assert crash_f.shape == (CRASH_POP_SIZE, crash_problem.n_obj)
    assert np.isfinite(crash_x).all() and np.isfinite(crash_f).all()
    assert np.all(crash_x >= crash_problem.lower)
    assert np.all(crash_x <= crash_problem.upper)
    np.testing.assert_allclose(crash_problem.evaluate(crash_x), crash_f, rtol=1e-12, atol=1e-12)
    dominates = (np.all(crash_f[:, None, :] <= crash_f[None, :, :], axis=2)
                 & np.any(crash_f[:, None, :] < crash_f[None, :, :], axis=2))
    crash_nondominated = ~dominates.any(axis=0)
    distinct_count = len(np.unique(crash_x, axis=0))
    print(f'Distinct designs: {distinct_count}/{CRASH_POP_SIZE}')
    print(f'Non-dominated in final population: {crash_nondominated.sum()}/{CRASH_POP_SIZE}')
    for label, values in zip(('f1 weight', 'f2 acceleration', 'f3 intrusion'), crash_f.T):
        print(f'{label}: min={values.min():.6g}, median={np.median(values):.6g}, max={values.max():.6g}')

    # 3D objective plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(crash_f[:, 0], crash_f[:, 1], crash_f[:, 2],
               s=28, c='0.40', edgecolors='black', linewidths=0.35, alpha=0.9)
    ax.set_xlabel(r'$f_1$ (weight)', labelpad=8)
    ax.set_ylabel(r'$f_2$ (acceleration)', labelpad=8)
    ax.set_zlabel(r'$f_3$ (toe-board intrusion)', labelpad=8)
    ax.set_title('NSGA-II: vehicle crashworthiness final population')
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    plt.show()

    # Pairwise objective plots
    pairs = ((0, 1), (0, 2), (1, 2))
    labels = (r'$f_1$ (weight)', r'$f_2$ (acceleration)',
              r'$f_3$ (toe-board intrusion)')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(crash_f[:, i], crash_f[:, j], s=20, c='0.40', alpha=0.85)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(alpha=0.25)
    fig.suptitle('Pairwise views of the same final population')
    fig.tight_layout()
    plt.show()

    return crash_x, crash_f, crash_runtime_seconds
