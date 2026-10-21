"""Given plant, experiment drivers, plotting, and file output for HW5.

The exercise equations are supplied by functions in the problem scripts.
All state and parameter histories store one time sample per row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HORIZON = 100
NOMINAL_PARAMETERS = np.array([0.1, 0.005, 0.1])
TRUE_PARAMETERS = np.array([0.12, 0.008, 0.11])
INITIAL_MEAN = np.array([2.0, 0.0])
INITIAL_STATE_COVARIANCE = 0.001 * np.eye(2)
INITIAL_PARAMETER_COVARIANCE = np.eye(3)
Q = np.eye(2)
R = np.array([[0.1]])
INITIAL_LEARNING_GAIN = 0.01 * np.eye(3)
FORGETTING_FACTOR = 1.0
PROCESS_COVARIANCE = 0.0001 * np.eye(2)
MEASUREMENT_COVARIANCE = 0.001 * np.eye(2)
KAPPA = 2.0


def plant_matrices(parameters):
    """Construct the given two-state model from [a, b, c]."""
    a, b, c = np.asarray(parameters, dtype=float)
    return np.array([[1.0, a], [0.0, 1.0]]), np.array([[b], [c]])


def parser(description, output_dir, input_dir=None, stochastic=False):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--output-dir", type=Path, default=output_dir)
    if input_dir is not None:
        result.add_argument("--input-dir", type=Path, default=input_dir,
                            help="Directory containing your Question 1.1 results.")
    if stochastic:
        result.add_argument("--seed", type=int, default=0,
                            help="Seed for the initial state and noise sequences (default: 0).")
    return result


def output_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_results(directory, filename):
    path = Path(directory) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Complete Question 1.1 or set --input-dir.")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def save_csv(path, columns, values):
    np.savetxt(path, values, delimiter=",", header=",".join(columns), comments="")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_figure(fig, out, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def require_array(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or np.iscomplexobj(value) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real array with shape {shape}.")
    return value.astype(float, copy=True)


def require_covariance(value, dimension, name):
    value = require_array(value, (dimension, dimension), name)
    scale = max(1.0, float(np.linalg.norm(value, ord=2)))
    if np.max(abs(value - value.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric.")
    symmetric = (value + value.T) / 2
    if np.linalg.eigvalsh(symmetric).min() < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite.")
    return symmetric


def simulate_mrac(a_star, error_function, update_function, gain_function):
    """Run 100 deterministic steps using the submitted adaptation functions."""
    a_star = require_array(a_star, (2, 2), "Reference matrix")
    a_true, b_true = plant_matrices(TRUE_PARAMETERS)
    a_initial, b_initial = plant_matrices(NOMINAL_PARAMETERS)
    theta = np.hstack([a_initial, b_initial])
    learning = INITIAL_LEARNING_GAIN.copy()
    states, reference = np.zeros((HORIZON + 1, 2)), np.zeros((HORIZON + 1, 2))
    states[0] = reference[0] = INITIAL_MEAN
    controls = np.zeros(HORIZON)
    models = np.zeros((HORIZON, 2, 3))
    learning_history = np.zeros((HORIZON, 3, 3))
    gains = np.zeros((HORIZON, 1, 2))
    for k in range(HORIZON):
        if k > 0:
            error = require_array(error_function(states[k], reference[k]), (2,), "Output error")
            regressor = np.concatenate([states[k-1], [controls[k-1]]])
            theta, learning = update_function(theta, learning, error, regressor, FORGETTING_FACTOR)
            theta = require_array(theta, (2, 3), "Estimated model")
            learning = require_covariance(learning, 3, "Learning matrix")
            if not np.allclose(theta[:, :2][[0, 1, 1], [0, 0, 1]], [1, 0, 1],
                               rtol=0, atol=1e-12):
                raise ValueError("The known entries of the estimated A matrix must stay fixed.")
        models[k], learning_history[k] = theta, learning
        gains[k] = require_array(gain_function(theta[:, :2], theta[:, 2:], a_star),
                                 (1, 2), "Matching gain")
        controls[k] = -(gains[k] @ states[k]).item()
        states[k+1] = a_true @ states[k] + b_true[:, 0] * controls[k]
        reference[k+1] = a_star @ reference[k]
    require_array(states, (HORIZON + 1, 2), "State history")
    errors = np.array([error_function(x, target) for x, target in zip(states, reference)])
    errors = require_array(errors, (HORIZON + 1, 2), "Tracking errors")
    return dict(states=states, reference=reference, errors=errors, controls=controls,
                parameters=models[:, [0, 0, 1], [1, 2, 2]], models=models,
                learning_matrices=learning_history, gains=gains,
                true_parameters=TRUE_PARAMETERS.copy(), A_star=a_star)


def save_mrac(result, out):
    np.savez_compressed(out / "mrac_trajectory.npz", **result)
    save_csv(out / "states.csv", ("k", "x1", "x2", "reference_x1", "reference_x2",
                                  "error_x1", "error_x2"),
             np.column_stack([np.arange(HORIZON + 1), result["states"],
                              result["reference"], result["errors"]]))
    save_csv(out / "controls.csv", ("k", "u", "K1", "K2"),
             np.column_stack([np.arange(HORIZON), result["controls"], result["gains"][:, 0]]))
    save_csv(out / "parameters.csv", ("k", "a_hat", "b_hat", "c_hat"),
             np.column_stack([np.arange(HORIZON), result["parameters"]]))
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    for component, style in enumerate(("-", "--")):
        axes[0].plot(result["reference"][:, component], "k" + style,
                     label=f"Reference x{component+1}")
        axes[0].plot(result["states"][:, component], "r" + style,
                     label=f"Actual x{component+1}")
    axes[0].set(ylabel="State", title="Model reference adaptive control")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(result["controls"], "k")
    axes[1].set_ylabel("Control")
    for component, name in enumerate(("a", "b", "c")):
        axis = axes[component + 2]
        axis.axhline(TRUE_PARAMETERS[component], color="k", label="True")
        axis.plot(result["parameters"][:, component], "r", label="Estimated")
        axis.set_ylabel(name)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Time step k")
    save_figure(fig, out, "mrac_trajectory")


def stochastic_inputs(seed):
    """Draw the PDF's random initial state and independent process/measurement noise."""
    rng = np.random.default_rng(seed)
    initial_state = INITIAL_MEAN + np.linalg.cholesky(INITIAL_STATE_COVARIANCE) @ rng.standard_normal(2)
    process_noise = rng.standard_normal((HORIZON, 2)) @ np.linalg.cholesky(PROCESS_COVARIANCE).T
    measurement_noise = rng.standard_normal((HORIZON, 2)) @ np.linalg.cholesky(MEASUREMENT_COVARIANCE).T
    return initial_state, process_noise, measurement_noise


def simulate_filter(initializer, lqr_function, filter_function, seed):
    """Run the plant with estimated-model LQR and a submitted filter step.

    The time-0 estimate is the supplied prior. The first measurement update
    uses y_1 after the first control and plant step.
    """
    mean, covariance, process_covariance, measurement_matrix = initializer(
        INITIAL_MEAN.copy(), INITIAL_STATE_COVARIANCE.copy(), NOMINAL_PARAMETERS.copy(),
        INITIAL_PARAMETER_COVARIANCE.copy(), PROCESS_COVARIANCE.copy())
    mean = require_array(mean, (5,), "Initial augmented mean")
    covariance = require_covariance(covariance, 5, "Initial augmented covariance")
    process_covariance = require_covariance(process_covariance, 5, "Augmented process covariance")
    measurement_matrix = require_array(measurement_matrix, (2, 5), "Measurement matrix")
    initial_state, process_noise, measurement_noise = stochastic_inputs(seed)
    a_true, b_true = plant_matrices(TRUE_PARAMETERS)
    states = np.zeros((HORIZON + 1, 2))
    estimates = np.zeros((HORIZON + 1, 5))
    covariances = np.zeros((HORIZON + 1, 5, 5))
    controls, gains = np.zeros(HORIZON), np.zeros((HORIZON, 1, 2))
    measurements = np.zeros((HORIZON, 2))
    states[0], estimates[0], covariances[0] = initial_state, mean, covariance
    for k in range(HORIZON):
        a_hat, b_hat = plant_matrices(estimates[k, 2:])
        try:
            _, gain, _ = lqr_function(a_hat, b_hat, Q, R)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"LQR solve failed at time {k} for estimates {estimates[k, 2:]}.") from exc
        gains[k] = require_array(gain, (1, 2), "Feedback gain")
        controls[k] = -(gains[k] @ estimates[k, :2]).item()
        states[k+1] = a_true @ states[k] + b_true[:, 0] * controls[k] + process_noise[k]
        measurements[k] = states[k+1] + measurement_noise[k]
        estimate, covariance = filter_function(
            estimates[k], covariances[k], controls[k], measurements[k],
            process_covariance, MEASUREMENT_COVARIANCE, measurement_matrix)
        estimates[k+1] = require_array(estimate, (5,), f"Estimate at time {k+1}")
        covariances[k+1] = require_covariance(covariance, 5, f"Covariance at time {k+1}")
    require_array(states, (HORIZON + 1, 2), "State history")
    return dict(states=states, estimates=estimates, covariances=covariances,
                controls=controls, gains=gains, measurements=measurements,
                process_noise=process_noise, measurement_noise=measurement_noise,
                true_parameters=TRUE_PARAMETERS.copy(), initial_mean=mean,
                initial_covariance=covariances[0].copy(), process_covariance=process_covariance,
                measurement_matrix=measurement_matrix,
                measurement_covariance=MEASUREMENT_COVARIANCE.copy())


def save_filter(result, out, method, seed, problem):
    np.savez_compressed(out / "filter_trajectory.npz", **result)
    save_csv(out / "states.csv", ("k", "x1", "x2"),
             np.column_stack([np.arange(HORIZON + 1), result["states"]]))
    save_csv(out / "estimates.csv", ("k", "x1_hat", "x2_hat", "a_hat", "b_hat", "c_hat"),
             np.column_stack([np.arange(HORIZON + 1), result["estimates"]]))
    save_csv(out / "controls.csv", ("k", "u", "K1", "K2"),
             np.column_stack([np.arange(HORIZON), result["controls"], result["gains"][:, 0]]))
    save_csv(out / "measurements.csv", ("k", "y1", "y2"),
             np.column_stack([np.arange(1, HORIZON + 1), result["measurements"]]))
    save_csv(out / "noise.csv", ("transition_k", "w1_k", "w2_k", "v1_next", "v2_next"),
             np.column_stack([np.arange(HORIZON), result["process_noise"],
                              result["measurement_noise"]]))
    names = ("x1", "x2", "a", "b", "c")
    save_csv(out / "covariance_diagonal.csv", ("k",) + names,
             np.column_stack([np.arange(HORIZON + 1),
                              np.diagonal(result["covariances"], axis1=1, axis2=2)]))
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    for component, axis in enumerate(axes):
        if component < 2:
            axis.plot(result["states"][:, component], "k", label="True")
        else:
            axis.axhline(TRUE_PARAMETERS[component - 2], color="k", label="True")
        axis.plot(result["estimates"][:, component], "r--", label="Estimated")
        axis.set_ylabel(names[component])
        axis.legend(fontsize=8)
    axes[0].set_title(f"{method}: simultaneous state and parameter estimation")
    axes[-1].set_xlabel("Time step k")
    save_figure(fig, out, "filter_trajectory")
    fig, axis = plt.subplots(figsize=(9, 3))
    axis.plot(result["controls"], "k")
    axis.set(xlabel="Time step k", ylabel="Control", title=f"{method}: estimated-model LQR")
    save_figure(fig, out, "control")
    summary = dict(
        problem=problem, method=method, seed=seed, executed_steps=HORIZON,
        true_initial_state=result["states"][0].tolist(),
        initial_estimate=result["initial_mean"].tolist(),
        initial_covariance=result["initial_covariance"].tolist(),
        true_parameters=TRUE_PARAMETERS.tolist(),
        final_state=result["states"][-1].tolist(),
        final_parameter_estimates=result["estimates"][-1, 2:].tolist(),
        state_estimation_rmse=np.sqrt(np.mean(
            (result["states"] - result["estimates"][:, :2])**2, axis=0)).tolist(),
        minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(result["covariances"]).min()),
        state_samples=HORIZON + 1, estimate_samples=HORIZON + 1,
        control_samples=HORIZON, first_measurement_used=1)
    if method == "UKF":
        summary["kappa"] = KAPPA
    save_json(out / "summary.json", summary)
