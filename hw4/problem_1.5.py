#!/usr/bin/env python3
"""Student scaffold for 16-714 HW4, Question 1.5: LQR feedback gain.

Keep hw4_helpers.py and hw4_inputs.py beside this script. Solvers, plotting,
and file output are provided; the functions below implement the exercise.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hw4_helpers as helpers

OUTPUT_DIR = HERE / "results" / "student_problem_1_5"


# %% Your implementation
def lqr_feedback(a, b, q, r):
    """Return the state cost matrix P, feedback gain K, and closed-loop matrix A_cl.

    Use the control convention u = -K x."""
    # TODO(student, Problem 1.5): Implement lqr_feedback using your derivation.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(a.shape[0], -1)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float).reshape(b.shape[1], b.shape[1])

    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(
        r + b.T @ p @ b,
        b.T @ p @ a,
    )
    a_cl = a - b @ k

    return p, k, a_cl


# %% Provided experiment and output

def solve():
    return lqr_feedback(helpers.A, helpers.B, helpers.Q, helpers.R)


def main():
    args = helpers.parser(__doc__, OUTPUT_DIR).parse_args()
    p, k, a_cl = solve()
    out = helpers.output_directory(args.output_dir)
    np.savez_compressed(out / "lqr_gain.npz", P=p, K=k, A_cl=a_cl)
    for name, value in [("P", p), ("K", k), ("A_cl", a_cl)]:
        helpers.save_csv(out / f"{name}.csv", ("position", "velocity"), value)
    helpers.save_json(out / "summary.json", {"problem": "1.5", "P": p.tolist(),
                                            "K": k.tolist(), "A_cl": a_cl.tolist()})
    print(f"K = {k}")
    print(f"Saved Question 1.5 results to {out}")


if __name__ == "__main__":
    main()
