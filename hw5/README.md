# Homework 5: adaptive control and state/parameter estimation

This code accompanies the HW5 handout on model reference adaptive control and
simultaneous state and parameter estimation. Use the handout for the
derivations and written discussions. These scripts implement its numerical
experiments with the stated parameter values.

Keep these files together:

```text
README.md
requirements.txt
hw5_helpers.py
problem_1.1.py
problem_1.5.py
problem_2.2.py
problem_2.3.py
```

## Setup

Use Python 3.10 or newer with NumPy, SciPy, and Matplotlib. From the
repository root, install the standalone dependencies:

```bash
python -m pip install -r hw5/requirements.txt
```

The experiments run headlessly without SPARK, MuJoCo, a GPU, or a viewer.

## Complete the exercises

Implement the functions marked `TODO(student, Problem ...)`. The helper
provides the given two-state plant, simulation loops, random input generation,
plotting, and file output. The assessed control and estimation equations are
implemented in the problem scripts.

| Script | Functions to complete |
|---|---|
| `problem_1.1.py` | `lqr_feedback` |
| `problem_1.5.py` | `tracking_error`, `parameter_update`, `matching_gain` |
| `problem_2.2.py` | `lqr_feedback`, `initial_filter`, `augmented_dynamics`, `dynamics_jacobian`, `ekf_predict`, `ekf_update` |
| `problem_2.3.py` | `lqr_feedback`, `initial_filter`, `augmented_dynamics`, `sigma_points`, `weighted_moments`, `cross_covariance`, `ukf_predict`, `ukf_update` |

Question 1.1 permits a library Riccati solver; `scipy.linalg.solve_discrete_are`
is already imported. Return the cost matrix, gain, and closed-loop matrix
using the convention `u = -K x`.

Question 1.5 uses your written derivations from Questions 1.2–1.4 and loads
your saved reference matrix from Question 1.1. Its parameter update uses the
output error from Question 1.2 and preserves the known entries of the
estimated state matrix.

Copy **your own completed** `lqr_feedback` function from Question 1.1 into
Questions 2.2 and 2.3. Use your Question 2.1 derivation to implement
`initial_filter` and `augmented_dynamics` in Question 2.2, then copy your own
implementations into Question 2.3. These functions start as TODOs in both
scripts. No earlier answers are supplied by later problem scripts.

The EKF state Jacobian treats the applied control as fixed during the
prediction. For the UKF, use the symmetric positive-semidefinite covariance
square root, kappa 2, and a new set of sigma points from the predicted
distribution for the measurement update. Function docstrings specify the
required array shapes.

Questions 1.2–1.4 and 2.1 require written work, which is used in the subsequent
scripts. Submit the discussions requested in Questions 1.5, 2.2, and 2.3 too.

## Parameters and timing

| Quantity | Value |
|---|---|
| Nominal / initial parameter estimates `[a, b, c]` | `[0.1, 0.005, 0.1]` |
| True parameters | `[0.12, 0.008, 0.11]` |
| State and control weights | `Q = I_2`, `R = 0.1` |
| Simulation length | 100 transitions |
| Question 1 initial state | `[2, 0]` |
| Question 1 initial learning matrix | `0.01 I_3` |
| Question 1 forgetting factor | `1` |
| Question 2 true initial state | Drawn from `N([2, 0], 0.001 I_2)` |
| Question 2 initial parameter uncertainty | `I_3` |
| State process-noise covariance | `0.0001 I_2` |
| Measurement-noise covariance | `0.001 I_2` |
| Parameter process noise | Zero |
| UKF kappa | `2` |

Question 2 keeps the true parameters fixed at the specified values. The
parameter uncertainty belongs to the filter's initial belief. The true
initial state and both noise sequences are independent random draws.

For Question 1.5, the initial estimate is used for the first control.
Adaptation begins at time 1 using the preceding state and input. State,
reference, and tracking-error histories have 101 rows (times 0–100);
parameter estimates, gains, and controls have 100 rows (times 0–99).

For Questions 2.2–2.3, the initial filter mean and covariance are the supplied
prior. The first control uses this mean. Each transition then produces a
new measurement and a filter update, starting with `y_1`. States and
augmented estimates have 101 rows (times 0–100), controls have 100 rows
(times 0–99), and measurements have 100 rows (times 1–100). Augmented
estimates use the ordering `[x1, x2, a, b, c]`.

## Run and save results

From the repository root, after completing the relevant TODOs:

```bash
python hw5/problem_1.1.py
python hw5/problem_1.5.py
python hw5/problem_2.2.py --seed 0
python hw5/problem_2.3.py --seed 0
```

An unfinished function raises `NotImplementedError` before writing results.
Importing a script does not run an experiment. Scripts also work when invoked
by absolute path from another directory.

Every script accepts `--output-dir PATH`. The default output directory for
Question 1.1 is `results/student_problem_1_1/`, with corresponding names for
the other questions. Question 1.5 accepts `--input-dir PATH` to select your
Question 1.1 results.

Both filtering scripts accept `--seed INTEGER` (default `0`). The same seed
gives EKF and UKF the same true initial state and process/measurement noise
samples. Their measured trajectories can differ because their controllers
use different estimates. For another experiment without overwriting the
default results:

```bash
python hw5/problem_2.2.py --seed 1 --output-dir hw5/results/student_problem_2_2_seed1
python hw5/problem_2.3.py --seed 1 --output-dir hw5/results/student_problem_2_3_seed1
```

| Question | Generated outputs |
|---|---|
| 1.1 | `P`, `K`, and `A_star` in CSV/NPZ, plus a JSON summary |
| 1.5 | States, reference, errors, controls, gains, parameter and learning-matrix histories, a summary, and PNG/PDF plots |
| 2.2 / 2.3 | True states, estimates, full covariance histories, measurements, controls, gains, noise samples, a summary, and PNG/PDF plots |

NPZ files retain the full numerical arrays. CSV files expose the main
trajectories and covariance diagonals. The summaries report final estimates
and estimation errors for interpreting the handout's discussion questions.
Stochastic plots vary with the seed; parameter estimates need not equal the
true values after 100 steps.

The dots in filenames are intentional; invoke scripts by file path. Use
ordinary `python` on macOS as well. Generated `hw5/results/` and locally
assembled `hw5/release/` packages are ignored by Git. Completed homework
implementations, written solutions, grading rubrics, and reference outputs
are excluded from this release.

## Check the installation

Use `--help` to check imports and command-line options without running an
exercise or writing results:

```bash
python hw5/problem_1.1.py --help
python hw5/problem_1.5.py --help
python hw5/problem_2.2.py --help
python hw5/problem_2.3.py --help
```

If Question 1.5 reports a missing `reference_model.npz`, complete and run
Question 1.1 first, or select its saved results using `--input-dir PATH`.
No files from another course checkout are required.
