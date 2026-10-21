#!/usr/bin/env python3
"""Student scaffold for 16-714 HW5, Question 2.3: UKF parameter adaptation.

Reuse your LQR function from Question 1.1 and augmented model from Question 2.1.
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

OUTPUT_DIR = HERE / "results" / "student_problem_2_3"


# %% Your implementation
def lqr_feedback(a, b, q, r):
    """Reuse your Question 1.1 function, returning P, K, and the closed-loop matrix."""
    # TODO(student, Problem 2.3): Implement lqr_feedback using your derivation.
    raise NotImplementedError("Implement Problem 2.3 lqr_feedback")


def initial_filter(state_mean, state_covariance, parameter_mean, parameter_covariance,
                   process_covariance):
    """Reuse your Question 2.1 initialization from Question 2.2.

    Return the augmented mean, covariance, process covariance, and measurement matrix
    with shapes (5,), (5,5), (5,5), and (2,5), using [x1, x2, a, b, c].
    """
    # TODO(student, Problem 2.3): Implement initial_filter using your derivation.
    raise NotImplementedError("Implement Problem 2.3 initial_filter")


def augmented_dynamics(estimate, control):
    """Reuse your noise-free augmented dynamics from Question 2.2; return shape (5,)."""
    # TODO(student, Problem 2.3): Implement augmented_dynamics using your derivation.
    raise NotImplementedError("Implement Problem 2.3 augmented_dynamics")


def sigma_points(mean, covariance, kappa):
    """Return a (2n+1,n) sigma-point array and (2n+1,) weights for the given kappa.

    Use the symmetric positive-semidefinite matrix square root of the covariance.
    """
    # TODO(student, Problem 2.3): Implement sigma_points using your derivation.
    raise NotImplementedError("Implement Problem 2.3 sigma_points")


def weighted_moments(points, weights):
    """Return the weighted mean and covariance of row-wise sigma points."""
    # TODO(student, Problem 2.3): Implement weighted_moments using your derivation.
    raise NotImplementedError("Implement Problem 2.3 weighted_moments")


def cross_covariance(state_points, measurement_points, weights):
    """Return the weighted state/measurement cross covariance."""
    # TODO(student, Problem 2.3): Implement cross_covariance using your derivation.
    raise NotImplementedError("Implement Problem 2.3 cross_covariance")


def ukf_predict(estimate, covariance, control, process_covariance, kappa):
    """Return the next prior mean and covariance using propagated sigma points."""
    # TODO(student, Problem 2.3): Implement ukf_predict using your derivation.
    raise NotImplementedError("Implement Problem 2.3 ukf_predict")


def ukf_update(prior, prior_covariance, measurement, measurement_matrix,
               measurement_covariance, kappa):
    """Return the posterior using new sigma points from the prior mean and covariance."""
    # TODO(student, Problem 2.3): Implement ukf_update using your derivation.
    raise NotImplementedError("Implement Problem 2.3 ukf_update")


# %% Provided experiment and output
def filter_step(estimate, covariance, control, measurement, process_covariance,
                measurement_covariance, measurement_matrix):
    prior, prior_covariance = ukf_predict(
        estimate, covariance, control, process_covariance, helpers.KAPPA)
    return ukf_update(prior, prior_covariance, measurement, measurement_matrix,
                      measurement_covariance, helpers.KAPPA)


def solve(seed=0):
    return helpers.simulate_filter(initial_filter, lqr_feedback, filter_step, seed)


def main():
    args = helpers.parser(__doc__, OUTPUT_DIR, stochastic=True).parse_args()
    result = solve(args.seed)
    out = helpers.output_directory(args.output_dir)
    helpers.save_filter(result, out, "UKF", args.seed, "2.3")
    print(f"Final parameter estimates: {result['estimates'][-1, 2:]}")
    print(f"Saved Question 2.3 results to {out}")


if __name__ == "__main__":
    main()
