## 4. Task 4: Hydro-static Thrust Bearing Design (RCM20)

**Model.** RCM20 has four variables (R, R0, μ, Q), two objectives and seven
inequality constraints [1]. We implemented it from the authors' repository [2]
(`CEC2021_func.m`, `func == 20`), keeping all constants and the three 10⁻⁵
correction terms, and checked it against an independent Python transcription of
the MATLAB lines on 200,000 random points. The supplementary equations differ
from this code: for example, P is printed as
(log₁₀log₁₀(8.122×10⁶μ + 0.8) + 3.55)/10.04 instead of
(log₁₀log₁₀(8.122×10⁶μ + 0.8) − 10.04)/(−3.55), the pressure constraint is
reversed (P0 ≥ 1000 instead of P0 ≤ 1000), and the temperature constraint bounds
P0 instead of ΔT. As printed, the model gives P0 < 0 wherever its formulas are
defined, contradicting P0 ≥ 1000, so it has no feasible solution. Following the
TA's advice, we used the repository code.

**Method.** We reused the Task 3 constrained NSGA-II unchanged, with
CV = Σ max(g_j, 0) and feasibility CV ≤ 10⁻¹². Undefined values (e.g. at
R = R0) get a violation larger than any defined point, very large values are
capped to avoid overflow, and all reported solutions were re-checked with the
uncapped formulas. From the baseline N = 100, G = 500, pc = 0.9, pm = 0.25 (per
variable), Stage A changed one parameter at a time; Stage B kept approximately
200,000 evaluations (initial population included) and changed only the N × G
split; Stage C combined the two best levels of N, pc and pm at exactly 200,000
evaluations. Each setting used five seeds. Runs were compared by hypervolume
(HV) with one fixed normalisation and reference point, where a larger HV is
better; the values are not comparable with other papers. A rule fixed in
advance chose the configuration, which was then tested on ten new seeds
(Table 1).

**Results.** Mutation had the clearest effect (Fig. 1): with pm = 0, three of
five runs found no feasible solution, while pm = 0.1–0.5 gave median HV
0.68–0.72. Crossover showed no clear trend in Stage A; in Stage C its effect
ranged from −0.041 to +0.108 depending on N and pm. Larger N and G improved HV in
Stage A but also used more evaluations; at about the same budget, the N × G
splits gave median HV 0.76–0.85 with overlapping ranges. The selected
configuration (N = 400, G = 499, pc = 0.9, pm = 0.25) had a higher HV than the
baseline parameters at equal budget on 7 of 10 new seeds. In an additional test
on seeds 0–4, its median HV was 0.710, 0.850 and 0.887 at 50,000, 200,000 and
800,000 evaluations.

**Final solutions and limitations.** The ten new-seed runs of the selected
configuration gave 908 non-dominated records, i.e. 598 distinct approximate
non-dominated solutions (Fig. 2), all within bounds and feasible under the
original formulas. Most lie close to the load, pressure and temperature limits,
which we observed but did not prove for the true Pareto front. A
known feasible point, (f1, f2) = (1621.31, 3.64051×10⁻⁶), dominates our
lowest-f1 solution (1646.06, 3.68457×10⁻⁶), so that end of the set is not fully
converged. Nine of the 195 runs found no feasible solution, and parameter
interactions were tested only in Stage C.

**Table 1.** Runs on seeds 100–109 (medians over ten seeds).

| configuration | evaluations | runs with a feasible solution | HV | distinct solutions | largest gap | run-minimum f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected: N = 400, G = 499, pc = 0.9, pm = 0.25 | 200,000 | 10/10 | 0.839 | 261.5 | 0.136 | 1657.75 |
| baseline parameters, N = 100, G = 1999 | 200,000 | 9/10 | 0.740 | 86.0 | 0.051 | 1634.35 |
| untuned baseline, N = 100, G = 500 | 50,100 | 9/10 | 0.667 | 86.5 | 0.178 | 1736.48 |

Runs without a feasible solution count as HV = 0. Distinct solutions are the
feasible non-dominated members of a run after removing identical decision
vectors. The largest gap is the largest distance between neighbouring distinct
solutions in normalised objective space; it describes spacing within the set,
not coverage of the whole front. The largest gap and the run-minimum f1 are
medians over runs with a feasible solution (nine runs in the two baseline rows).

**Fig. 1.** Stage A: one parameter varied around the baseline (N = 100, G = 500,
pc = 0.9, pm = 0.25), five seeds per level. Line: median HV; band: minimum to
maximum over seeds; purple: share of runs with a feasible solution; green:
median share of feasible individuals in the final population. Changing N or G
also changes the number of evaluations.

**Fig. 2.** Selected configuration. Left: final population of seed 103, the
lower-median run by HV among seeds 100–109. Right: the 598 distinct solutions
pooled from seeds 100–109; the cross marks a known feasible point that was not
produced by the optimiser.

**References**

[1] A. Kumar, G. Wu, M. Z. Ali, Q. Luo, R. Mallipeddi, P. N. Suganthan and
S. Das, "A benchmark-suite of real-world constrained multi-objective
optimization problems and some baseline results," *Swarm and Evolutionary
Computation*, vol. 67, art. 100961, 2021, doi: 10.1016/j.swevo.2021.100961.

[2] P. N. Suganthan, "2021-RW-MOP," GitHub repository,
https://github.com/P-N-Suganthan/2021-RW-MOP, file `CEC2021_func.m` (`func == 20`)
in `CEC2021-RWCMOP.zip`, accessed 23 Oct. 2026.
