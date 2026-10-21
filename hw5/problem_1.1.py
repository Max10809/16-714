#!/usr/bin/env python3
"""Student scaffold for 16-714 HW5, Question 1.1: nominal LQR reference model.

Keep hw5_helpers.py beside this script.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hw5_helpers as helpers

OUTPUT_DIR = HERE / "results" / "student_problem_1_1"


# %% Your implementation
def lqr_feedback(a, b, q, r):
    """Return P, K, and A_star for the convention u = -K x.

    A library Riccati solver is permitted in Question 1.1.
    """
    # TODO(student, Problem 1.1): Implement lqr_feedback using your derivation.
    raise NotImplementedError("Implement Problem 1.1 lqr_feedback")


# %% Provided experiment and output
def solve():
    a, b = helpers.plant_matrices(helpers.NOMINAL_PARAMETERS)
    return lqr_feedback(a, b, helpers.Q, helpers.R)


def main():
    args = helpers.parser(__doc__, OUTPUT_DIR).parse_args()
    p, gain, a_star = solve()
    out = helpers.output_directory(args.output_dir)
    np.savez_compressed(out / "reference_model.npz", P=p, K=gain, A_star=a_star)
    for name, values in (("P", p), ("K", gain), ("A_star", a_star)):
        helpers.save_csv(out / f"{name}.csv", ("x1", "x2"), values)
    helpers.save_json(out / "summary.json", {
        "problem": "1.1", "P": p.tolist(), "K": gain.tolist(), "A_star": a_star.tolist()})
    print(f"K = {gain}")
    print(f"A_star =\n{a_star}")
    print(f"Saved Question 1.1 results to {out}")


if __name__ == "__main__":
    main()
