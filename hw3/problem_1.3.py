#!/usr/bin/env python3
"""Student scaffold for 16-714 HW3, Question 1.3: maximal controlled invariant set.

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

OUTPUT_DIR = HERE / "results" / "student_problem_1_3"


# %% Mathematical implementation
def predecessor_intersection(matrix, bound, a, b, state_bound, input_bound):
    """Return inequalities for X intersect Pre(C), where C = {x: matrix x <= bound}.

    Construct joint inequalities in [position, velocity, u] using C's
    successor-state bounds and |u| <= input_bound. The provided generic
    eliminate_scalar helper projects out u. Finally intersect with X.
    """
    # TODO(student, Problem 1.3): Form the constrained predecessor and intersect it with the state box.
    matrix = np.asarray(matrix, dtype=float)
    bound = np.asarray(bound, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float)

    n = a.shape[0]
    b = np.asarray(b, dtype=float).reshape(n, 1)

    state_bound = np.broadcast_to(
    np.asarray(state_bound, dtype=float).reshape(-1), (n,)
    )
    input_bound = float(np.asarray(input_bound).item())

    successor_matrix = np.hstack([
        matrix @ a,
        matrix @ b,
    ])
    
    input_matrix = np.zeros((2, n + 1))
    input_matrix[0, -1] = 1.0
    input_matrix[1, -1] = -1.0

    joint_matrix = np.vstack([
        successor_matrix,
        input_matrix,
    ])
    joint_bound = np.concatenate([
        bound,
        [input_bound, input_bound],
    ])

    # Project out u to obtain Pre(C).
    pre_matrix, pre_bound = helpers.eliminate_scalar(
        joint_matrix, joint_bound
    )

    # Intersect Pre(C) with X = {x: -state_bound <= x <= state_bound}.
    new_matrix = np.vstack([
        pre_matrix,
        np.eye(n),
        -np.eye(n),
    ])
    new_bound = np.concatenate([
        np.asarray(pre_bound).reshape(-1),
        state_bound,
        state_bound,
    ])

    return new_matrix, new_bound


# %% Provided fixed-point iteration, plotting, and saving
def solve():
    return helpers.invariant_iteration(predecessor_intersection)


def main():
    args = helpers.parser(__doc__, OUTPUT_DIR).parse_args()
    matrix, bound, vertices, history, fixed_point_error = solve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    helpers.save_csv(out / "vertices.csv", ("position", "velocity"), vertices)
    helpers.save_csv(out / "halfspaces.csv", ("position_coefficient", "velocity_coefficient", "bound"),
                     np.column_stack([matrix, bound]))
    helpers.save_csv(out / "iterations.csv", ("iteration", "vertices", "area"), history)
    np.savez_compressed(out / "invariant_set.npz", H=matrix, h=bound, vertices=vertices,
                        iteration_history=history)
    helpers.plot_invariant(vertices, out)
    helpers.save_json(out / "summary.json", {
        "problem": "1.3", "method": "controlled-predecessor fixed point from the state box",
        "iterations": int(history[-1, 0]), "vertices": len(vertices),
        "area": helpers.polygon_area(vertices), "fixed_point_tolerance": 1e-9,
        "fixed_point_containment_error": fixed_point_error,
        "figure": "invariant_set.png"})
    print(f"Question 1.3: fixed point after {int(history[-1,0])} iterations; {len(vertices)} vertices")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
