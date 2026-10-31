#!/usr/bin/env python3
"""Student scaffold for 16-714 HW4, Question 2.5: state estimation and feedback simulation.

Keep hw4_helpers.py and hw4_inputs.py beside this script. Solvers, plotting,
and file output are provided; the functions below implement the exercise.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hw4_helpers as helpers

OUTPUT_DIR = HERE / "results" / "student_problem_2_5"
CONTROL_DIR = HERE / "results" / "student_problem_2_2"
FILTER_DIR = HERE / "results" / "student_problem_2_4"


# %% Your implementation
def initial_distribution(initial_state, state_covariance, disturbance_variance):
    """Return the augmented initial mean and covariance from Question 2.1."""
    # TODO(student, Problem 2.5): Implement initial_distribution using your derivation.
    initial_state = np.asarray(initial_state, dtype=float).reshape(2)
    state_covariance = np.asarray(state_covariance, dtype=float)

    mean = np.concatenate([initial_state, [0.0]])
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = state_covariance
    covariance[2, 2] = disturbance_variance

    return mean, covariance


def feedback_control(estimate, gain):
    """Return the scalar control using the augmented posterior state estimate."""
    # TODO(student, Problem 2.5): Implement feedback_control using your derivation.
    estimate = np.asarray(estimate, dtype=float).reshape(3)
    gain = np.asarray(gain, dtype=float).reshape(1, 3)

    return -(gain @ estimate).item()



def kalman_step(previous_estimate, previous_control, measurement, a, b, c, gain):
    """Return the updated estimate after prediction and the new scalar measurement.

    The driver calls this for times 1 through 99. The time-0 estimate is the prior mean."""
    # TODO(student, Problem 2.5): Implement kalman_step using your derivation.
    previous_estimate = np.asarray(
        previous_estimate, dtype=float
    ).reshape(3)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).reshape(3)
    c = np.asarray(c, dtype=float).reshape(1, 3)
    gain = np.asarray(gain, dtype=float).reshape(3)

    predicted = a @ previous_estimate + b * previous_control
    innovation = float(measurement) - (c @ predicted).item()

    return predicted + gain * innovation


# %% Provided experiment and output

def solve(control_dir=CONTROL_DIR, filter_dir=FILTER_DIR, seed=None):
    gain = helpers.load_results(control_dir, "control_gain.npz")["K"]
    model = helpers.load_results(filter_dir, "kalman_filter.npz")
    return helpers.simulate_lqg(initial_distribution, feedback_control, kalman_step, gain, model, seed)


def main():
    parser = helpers.parser(__doc__, OUTPUT_DIR)
    parser.add_argument("--control-dir", type=Path, default=CONTROL_DIR)
    parser.add_argument("--filter-dir", type=Path, default=FILTER_DIR)
    helpers.add_seed_argument(parser)
    args = parser.parse_args()
    result = solve(args.control_dir, args.filter_dir, args.seed)
    out = helpers.output_directory(args.output_dir)
    helpers.save_lqg(result, out)
    helpers.save_json(out / "summary.json", {"problem": "2.5", "seed": args.seed,
        "reference_inputs": args.seed is None, "state_samples": len(result["states"]),
        "estimate_samples": len(result["estimates"]), "control_samples": len(result["controls"]),
        "first_measurement_used": 1, "initial_state": result["states"][0].tolist()})
    print(f"Saved Question 2.5 results to {out}")


if __name__ == "__main__":
    main()
