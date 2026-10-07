#!/usr/bin/env python3
"""Student scaffold for 16-714 HW3, Question 1.2: finite-horizon feasible sets.

Keep hw3_helpers.py beside this script. Simulation, solver calls, plots, and
artifact I/O are provided; the mathematical functions below are the exercise.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Provided sibling import; works when this script is run from any directory.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hw3_helpers as helpers

OUTPUT_DIR = HERE / "results" / "student_problem_1_2"


# %% Mathematical implementation
def lifted_dynamics(a, b, horizon):
    """Return F, G such that stacked states X = F @ x0 + G @ U.

    Stack x0 through x_N and u0 through u_(N-1), in time order.
    For n states and m inputs, return shapes ((N+1)*n,n) and ((N+1)*n,N*m).
    """
    # TODO(student, Problem 1.2): Implement the lifted dynamics derived in Question 1.1.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    n, m = b.shape
    f = np.zeros(((horizon + 1) * n, n))
    g = np.zeros(((horizon + 1) * n, horizon * m))

    # Initial state: x0 = I @ x0 + 0 @ U.
    f[:n, :] = np.eye(n)

    for i in range(1, horizon + 1):
        prev = slice((i - 1) * n, i * n)
        curr = slice(i * n, (i + 1) * n)

        # xi = A @ x_(i-1) + B @ u_(i-1).
        f[curr, :] = a @ f[prev, :]
        g[curr, :] = a @ g[prev, :]
        g[curr, (i - 1) * m:i * m] = b

    return f, g


def feasibility_constraints(state, f, g, state_bound, input_bound):
    """Return A_ineq, b_ineq for the condensed inequalities A_ineq U <= b_ineq.

    f: (2*(N+1),2); g: (2*(N+1),N); state: (2,).
    Include both sides of every state and input bound, including x0 and x_N.
    """
    # TODO(student, Problem 1.2): Construct the lifted state/input inequalities derived in Question 1.1.
    state = np.asarray(state, dtype=float).reshape(-1)
    n = state.size
    num_states = f.shape[0] // n
    num_controls = g.shape[1]

    # Repeat bounds for every predicted state and control.
    state_bound = np.broadcast_to(
        np.asarray(state_bound, dtype=float).reshape(-1), (n,)
    )
    bx = np.tile(state_bound, num_states)
    bu = np.full(num_controls, float(np.asarray(input_bound).item()))

    free_response = f @ state
    identity = np.eye(num_controls)

    a_ineq = np.vstack([
        g,
        -g,
        identity,
        -identity,
    ])
    b_ineq = np.concatenate([
        bx - free_response,
        bx + free_response,
        bu,
        bu,
    ])

    return a_ineq, b_ineq


# %% Provided grid sampling, plotting, and saving
def solve(resolution=0.1, horizons=(2, 10, 100)):
    masks = {}
    for horizon in horizons:
        grid, masks[horizon] = helpers.feasibility_grid(
            feasibility_constraints, lifted_dynamics, horizon, resolution)
        print(f"N={horizon}: {masks[horizon].sum()} feasible / {masks[horizon].size} sampled states", flush=True)
    return grid, masks


def main():
    parser = helpers.parser(__doc__, OUTPUT_DIR)
    parser.add_argument("--resolution", type=float, default=0.1,
                        help="State-grid spacing (assignment default: 0.1).")
    args = parser.parse_args()
    grid, masks = solve(args.resolution)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    position, velocity = np.meshgrid(grid, grid)
    for horizon, mask in masks.items():
        helpers.save_csv(out / f"feasible_N{horizon}.csv", ("position", "velocity", "feasible"),
                         np.column_stack([position.ravel(), velocity.ravel(), mask.ravel().astype(int)]))
    np.savez_compressed(out / "feasible_sets.npz", grid=grid,
                        **{f"N{horizon}": mask for horizon, mask in masks.items()})
    helpers.plot_feasible_sets(grid, masks, out)
    helpers.save_json(out / "summary.json", {
        "problem": "1.2", "method": "LP feasibility on the initial-state grid",
        "resolution": args.resolution, "state_order": ["position", "velocity"],
        "feasible_counts": {str(n): int(mask.sum()) for n, mask in masks.items()},
        "sample_count_per_horizon": int(len(grid)**2),
        "nested_sets": bool(np.all(~masks[100] | masks[10]) and np.all(~masks[10] | masks[2])),
        "figure": "feasible_sets.png"})
    print(f"Saved Question 1.2 results to {out}")


if __name__ == "__main__":
    main()
