#!/usr/bin/env python3
"""Student scaffold for 16-714 HW5, Question 2.2: EKF parameter adaptation.

Use the augmented model from Question 2.1 and reuse your LQR function from
Question 1.1. Keep hw5_helpers.py beside this script.
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

OUTPUT_DIR = HERE / "results" / "student_problem_2_2"


# %% Your implementation
def lqr_feedback(a, b, q, r):
    """Reuse your Question 1.1 function, returning P, K, and the closed-loop matrix."""
    # TODO(student, Problem 2.2): Implement lqr_feedback using your derivation.
    raise NotImplementedError("Implement Problem 2.2 lqr_feedback")


def initial_filter(state_mean, state_covariance, parameter_mean, parameter_covariance,
                   process_covariance):
    """Return the augmented mean, covariance, process covariance, and measurement matrix.

    Use your Question 2.1 derivation with ordering [x1, x2, a, b, c].
    Required shapes are (5,), (5,5), (5,5), and (2,5).
    """
    # TODO(student, Problem 2.2): Implement initial_filter using your derivation.
    raise NotImplementedError("Implement Problem 2.2 initial_filter")


def augmented_dynamics(estimate, control):
    """Return the noise-free augmented state update from Question 2.1, with shape (5,)."""
    # TODO(student, Problem 2.2): Implement augmented_dynamics using your derivation.
    raise NotImplementedError("Implement Problem 2.2 augmented_dynamics")


def dynamics_jacobian(estimate, control):
    """Return the (5,5) augmented state Jacobian, treating the applied input as fixed."""
    # TODO(student, Problem 2.2): Implement dynamics_jacobian using your derivation.
    raise NotImplementedError("Implement Problem 2.2 dynamics_jacobian")


def ekf_predict(estimate, covariance, control, process_covariance):
    """Return the next prior mean and covariance using your EKF prediction equations."""
    # TODO(student, Problem 2.2): Implement ekf_predict using your derivation.
    raise NotImplementedError("Implement Problem 2.2 ekf_predict")


def ekf_update(prior, prior_covariance, measurement, measurement_matrix, measurement_covariance):
    """Return the posterior mean and covariance using the new two-component measurement."""
    # TODO(student, Problem 2.2): Implement ekf_update using your derivation.
    raise NotImplementedError("Implement Problem 2.2 ekf_update")


# %% Provided experiment and output
def filter_step(estimate, covariance, control, measurement, process_covariance,
                measurement_covariance, measurement_matrix):
    prior, prior_covariance = ekf_predict(estimate, covariance, control, process_covariance)
    return ekf_update(prior, prior_covariance, measurement, measurement_matrix, measurement_covariance)


def solve(seed=0):
    return helpers.simulate_filter(initial_filter, lqr_feedback, filter_step, seed)


def main():
    args = helpers.parser(__doc__, OUTPUT_DIR, stochastic=True).parse_args()
    result = solve(args.seed)
    out = helpers.output_directory(args.output_dir)
    helpers.save_filter(result, out, "EKF", args.seed, "2.2")
    print(f"Final parameter estimates: {result['estimates'][-1, 2:]}")
    print(f"Saved Question 2.2 results to {out}")


if __name__ == "__main__":
    main()
