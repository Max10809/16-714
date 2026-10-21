#!/usr/bin/env python3
"""Student scaffold for 16-714 HW5, Question 1.5: model reference adaptation.

Implement the equations from Questions 1.2 and 1.4, using your saved reference
matrix from Question 1.1. Keep hw5_helpers.py beside this script.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hw5_helpers as helpers

OUTPUT_DIR = HERE / "results" / "student_problem_1_5"
INPUT_DIR = HERE / "results" / "student_problem_1_1"


# %% Your implementation
def tracking_error(state, reference_state):
    """Return the two-component output error defined in Question 1.2."""
    # TODO(student, Problem 1.5): Implement tracking_error using your derivation.
    raise NotImplementedError("Implement Problem 1.5 tracking_error")


def parameter_update(theta, learning, error, regressor, forgetting_factor):
    """Return the updated (2,3) model [A_hat B_hat] and (3,3) learning matrix.

    Use the tracking-error adaptation rule from Question 1.2 and retain the
    known entries of A_hat. The regressor contains the preceding state and input.
    """
    # TODO(student, Problem 1.5): Implement parameter_update using your derivation.
    raise NotImplementedError("Implement Problem 1.5 parameter_update")


def matching_gain(a_hat, b_hat, a_star):
    """Return the (1,2) least-squares matching gain derived in Question 1.4."""
    # TODO(student, Problem 1.5): Implement matching_gain using your derivation.
    raise NotImplementedError("Implement Problem 1.5 matching_gain")


# %% Provided experiment and output
def solve(input_dir=INPUT_DIR):
    a_star = helpers.load_results(input_dir, "reference_model.npz")["A_star"]
    return helpers.simulate_mrac(a_star, tracking_error, parameter_update, matching_gain)


def main():
    args = helpers.parser(__doc__, OUTPUT_DIR, INPUT_DIR).parse_args()
    result = solve(args.input_dir)
    out = helpers.output_directory(args.output_dir)
    helpers.save_mrac(result, out)
    helpers.save_json(out / "summary.json", {
        "problem": "1.5", "executed_steps": helpers.HORIZON,
        "initial_parameters": helpers.NOMINAL_PARAMETERS.tolist(),
        "true_parameters": helpers.TRUE_PARAMETERS.tolist(),
        "initial_learning_gain": helpers.INITIAL_LEARNING_GAIN.tolist(),
        "forgetting_factor": helpers.FORGETTING_FACTOR,
        "final_state": result["states"][-1].tolist(),
        "final_tracking_error": result["errors"][-1].tolist(),
        "final_parameter_estimates": result["parameters"][-1].tolist(),
        "state_samples": len(result["states"]),
        "parameter_samples": len(result["parameters"])})
    print(f"Final parameter estimates: {result['parameters'][-1]}")
    print(f"Saved Question 1.5 results to {out}")


if __name__ == "__main__":
    main()
