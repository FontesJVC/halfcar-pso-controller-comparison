"""
halfcar_pso_comparison.py

Reproducible comparison of three PSO-based state-feedback design strategies
for a four-DOF half-car active suspension:

    1. PSO-LQR    : PSO optimizes the LQR weighting parameters.
    2. PSO-PPC    : PSO optimizes desired closed-loop pole locations.
    3. PSO-Direct : PSO optimizes the entries of the feedback gain matrix K.

The script can:
- reproduce all independent PSO runs;
- resume an interrupted experiment without repeating completed runs;
- generate descriptive statistics;
- save the best controller obtained by each method in NPZ and human-readable JSON;
- generate all figures used for analysis;
- create both:
    * a boxplot with all three methods;
    * a focused PSO-Direct vs PSO-LQR boxplot for the manuscript;
- save the complete numerical configuration and software versions.

Typical use
-----------
1) Full reproduction:
       RUN_OPTIMIZATION = True
       N_RUNS = 30
   Then run the script.

2) Recreate tables/figures from an existing runs.csv:
       RUN_OPTIMIZATION = False
   The script will read OUTPUT_DIR / "runs.csv" and regenerate the outputs.

Dependencies
------------
numpy
pandas
matplotlib
scipy
control
pyswarms
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import trapezoid
from scipy.optimize import linear_sum_assignment
import control as ct
import pyswarms as ps


# =============================================================================
# 0. REPRODUCIBILITY / USER CONFIGURATION
# =============================================================================

# Repository-safe default: post-process the bundled runs.csv without rerunning PSO.
# Use --run-optimization for a complete new set of 90 optimization runs.
RUN_OPTIMIZATION = False

# Number of independent PSO runs per method.
N_RUNS = 30

# Seeds used are BASE_SEED, ..., BASE_SEED + N_RUNS - 1.
BASE_SEED = 1000

# Keep all three methods for the full study.
METHODS = ("LQR", "PPC", "DIRECT")

# When True, method/seed pairs already present in runs.csv are skipped.
RESUME = True

# PySwarms progress bar.
SHOW_PSO_PROGRESS = True

# Output folder is created beside this script when possible.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUTPUT_DIR = SCRIPT_DIR / "pso_repetitions_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV = OUTPUT_DIR / "runs.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
BEST_NPZ = OUTPUT_DIR / "best_controllers.npz"
BEST_JSON = OUTPUT_DIR / "best_controllers.json"
CONFIG_JSON = OUTPUT_DIR / "experiment_config.json"

BOXPLOT_ALL_PNG = OUTPUT_DIR / "cost_boxplot_all_methods.png"
BOXPLOT_FOCUSED_PNG = OUTPUT_DIR / "cost_boxplot_direct_lqr.png"
RUNPLOT_PNG = OUTPUT_DIR / "cost_by_run.png"
BEST_RESPONSE_PNG = OUTPUT_DIR / "best_responses.png"

# Also save PDF versions for vector-quality archival/repository use.
BOXPLOT_ALL_PDF = OUTPUT_DIR / "cost_boxplot_all_methods.pdf"
BOXPLOT_FOCUSED_PDF = OUTPUT_DIR / "cost_boxplot_direct_lqr.pdf"
RUNPLOT_PDF = OUTPUT_DIR / "cost_by_run.pdf"
BEST_RESPONSE_PDF = OUTPUT_DIR / "best_responses.pdf"

# PSO coefficients.
PSO_OPTIONS = {"c1": 0.5, "c2": 0.3, "w": 0.9}

# Method-specific optimization settings.
PSO_CONFIG = {
    "LQR": {"particles": 50, "dimensions": 10, "iters": 100},
    "PPC": {"particles": 50, "dimensions": 8, "iters": 100},
    "DIRECT": {"particles": 60, "dimensions": 16, "iters": 110},
}

# Search bounds.
LOWER_LQR = np.array([1e-3] * 8 + [1e-7] * 2, dtype=float)
UPPER_LQR = np.array([1e5] * 8 + [1e-2] * 2, dtype=float)

LOWER_PPC = np.ones(8, dtype=float) * 0.1
UPPER_PPC = np.array([30.0, 50.0] * 4, dtype=float)

K_LIMIT = 20000.0
LOWER_DIRECT = np.full(16, -K_LIMIT, dtype=float)
UPPER_DIRECT = np.full(16, K_LIMIT, dtype=float)

BOUNDS = {
    "LQR": (LOWER_LQR, UPPER_LQR),
    "PPC": (LOWER_PPC, UPPER_PPC),
    "DIRECT": (LOWER_DIRECT, UPPER_DIRECT),
}

# Stability requirement.
STABILITY_MARGIN = -1e-5

# Numerical penalty returned to PSO for invalid candidates.
PENALTY_COST = 1e15

# PPC verification:
# after pole placement, compare requested and achieved eigenvalues.
PPC_POLE_TOL = 1e-3


# =============================================================================
# 1. HALF-CAR MODEL
# =============================================================================

Lf, Lr = 1.27, 1.71
Km, Cm = 200000.0, 0.0

M = 1800.0
Iy = 3443.0

Mf, Mr = 90.0, 140.0

Cmf, Cmr = 1190.0, 1000.0
Kmf, Kmr = 66820.0, 18615.0

A = np.zeros((8, 8))

A[0, 1] = 1.0
A[2, 3] = 1.0
A[4, 5] = 1.0
A[6, 7] = 1.0

A[1, :] = [
    -(Kmr + Kmf) / M,
    -(Cmr + Cmf) / M,
    (Lr * Kmr - Lf * Kmf) / M,
    (Lr * Cmr - Lf * Cmf) / M,
    Kmr / M,
    Cmr / M,
    Kmf / M,
    Cmf / M,
]

A[3, :] = [
    (Lr * Kmr - Lf * Kmf) / Iy,
    (Lr * Cmr - Lf * Cmf) / Iy,
    -(Lr**2 * Kmr + Lf**2 * Kmf) / Iy,
    -(Lr**2 * Cmr + Lf**2 * Cmf) / Iy,
    -Lr * Kmr / Iy,
    -Lr * Cmr / Iy,
    Lf * Kmf / Iy,
    Lf * Cmf / Iy,
]

A[5, :] = [
    Kmr / Mr,
    Cmr / Mr,
    -Lr * Kmr / Mr,
    -Lr * Cmr / Mr,
    -(Km + Kmr) / Mr,
    -(Cm + Cmr) / Mr,
    0.0,
    0.0,
]

A[7, :] = [
    Kmf / Mf,
    Cmf / Mf,
    Lf * Kmf / Mf,
    Lf * Cmf / Mf,
    0.0,
    0.0,
    -(Km + Kmf) / Mf,
    -(Cm + Cmf) / Mf,
]

B = np.zeros((8, 2))
B[1, :] = [1.0 / M, 1.0 / M]
B[3, :] = [-Lr / Iy, Lf / Iy]
B[5, 0] = -1.0 / Mr
B[7, 1] = -1.0 / Mf

G = np.zeros((8, 2))
G[5, 0] = Km / Mr
G[7, 1] = Km / Mf

# Rows used to calculate body vertical and pitch accelerations.
C_ACCEL = A[[1, 3], :]
D_ACCEL = B[[1, 3], :]


# =============================================================================
# 2. ROAD DISTURBANCE / SIMULATION
# =============================================================================

DT = 0.01
T_SIM = np.arange(0.0, 10.0, DT)

VEHICLE_SPEED = 20.0
AXLE_DELAY = (Lf + Lr) / VEHICLE_SPEED

BUMP_HEIGHT = 0.1
BUMP_LENGTH = 5.0
BUMP_START_FRONT = 1.0
BUMP_DURATION = BUMP_LENGTH / VEHICLE_SPEED


def raised_cosine_bump(t, start, duration, height):
    """Raised-cosine road bump."""
    z = np.zeros_like(t, dtype=float)
    mask = (t >= start) & (t <= start + duration)

    z[mask] = 0.5 * height * (
        1.0 - np.cos(
            2.0 * np.pi * (t[mask] - start) / duration
        )
    )

    return z


# Disturbance convention used throughout the model:
# row 0 = rear road input
# row 1 = front road input
W = np.vstack(
    (
        raised_cosine_bump(
            T_SIM,
            BUMP_START_FRONT + AXLE_DELAY,
            BUMP_DURATION,
            BUMP_HEIGHT,
        ),
        raised_cosine_bump(
            T_SIM,
            BUMP_START_FRONT,
            BUMP_DURATION,
            BUMP_HEIGHT,
        ),
    )
)


# =============================================================================
# 3. MULTIOBJECTIVE COST
# =============================================================================

W_COMFORT = 1.0
W_STABILITY = 0.1
W_CONTROL = 1e-6
W_SMOOTHNESS = 1e-8


def calculate_cost(t, x, u, road_input, return_parts=False):
    """
    Evaluate the weighted objective used by all three design strategies.

    Components:
    - comfort: squared body vertical and pitch accelerations;
    - stability: squared front/rear tire deflections;
    - control effort: squared actuator forces;
    - smoothness: squared time derivative of actuator forces.
    """
    acceleration = C_ACCEL @ x + D_ACCEL @ u
    du = np.gradient(u, t, axis=1)

    raw_comfort = trapezoid(
        acceleration[0] ** 2 + acceleration[1] ** 2,
        t,
    )

    raw_stability = trapezoid(
        (x[4] - road_input[0]) ** 2
        + (x[6] - road_input[1]) ** 2,
        t,
    )

    raw_control = trapezoid(
        u[0] ** 2 + u[1] ** 2,
        t,
    )

    raw_smoothness = trapezoid(
        du[0] ** 2 + du[1] ** 2,
        t,
    )

    weighted_comfort = W_COMFORT * raw_comfort
    weighted_stability = W_STABILITY * raw_stability
    weighted_control = W_CONTROL * raw_control
    weighted_smoothness = W_SMOOTHNESS * raw_smoothness

    total_cost = (
        weighted_comfort
        + weighted_stability
        + weighted_control
        + weighted_smoothness
    )

    if not return_parts:
        return float(total_cost)

    return float(total_cost), {
        "comfort": float(weighted_comfort),
        "stability": float(weighted_stability),
        "control_effort": float(weighted_control),
        "smoothness": float(weighted_smoothness),
        "raw_comfort": float(raw_comfort),
        "raw_stability": float(raw_stability),
        "raw_control_effort": float(raw_control),
        "raw_smoothness": float(raw_smoothness),
    }


# =============================================================================
# 4. CONTROLLER PARAMETERIZATIONS
# =============================================================================

class PolePlacementMismatch(RuntimeError):
    """Raised when achieved PPC poles do not match the requested poles."""


def desired_ppc_poles(position):
    """
    Convert the 8-dimensional PPC parameter vector into four complex-conjugate
    pole pairs: -alpha_i +/- j beta_i.
    """
    p = np.asarray(position, dtype=float)

    poles = []

    for i in range(0, 8, 2):
        alpha = p[i]
        beta = p[i + 1]

        poles.extend(
            [
                complex(-alpha, beta),
                complex(-alpha, -beta),
            ]
        )

    return np.asarray(poles, dtype=complex)


def normalized_pole_matching_error(actual_poles, desired_poles):
    """
    Match actual and requested poles one-to-one using the Hungarian algorithm
    and return the largest normalized pole mismatch.
    """
    actual = np.asarray(actual_poles, dtype=complex)
    desired = np.asarray(desired_poles, dtype=complex)

    distance = np.abs(
        actual[:, None] - desired[None, :]
    )

    row_idx, col_idx = linear_sum_assignment(distance)

    matched_error = distance[row_idx, col_idx]
    scale = 1.0 + np.abs(desired[col_idx])

    return float(
        np.max(matched_error / scale)
    )


def closed_loop_eigenvalues(K):
    return np.linalg.eigvals(A - B @ K)


def maximum_real_eigenvalue(K):
    eigvals = closed_loop_eigenvalues(K)
    return float(np.max(np.real(eigvals)))


def controller_from_position(position, method, diagnostics=None):
    """
    Construct K from the optimization coordinates of each method.

    LQR:
        position = diag(Q) [8] + diag(R) [2]

    PPC:
        position = [alpha_1, beta_1, ..., alpha_4, beta_4]

    DIRECT:
        position = 16 entries of K, reshaped as 2 x 8
    """
    p = np.asarray(position, dtype=float)

    if method == "LQR":
        # Important:
        # no post-optimization clipping is performed. The same declared bounds
        # used by PSO are used consistently to construct the final controller.
        Q = np.diag(p[:8])
        R = np.diag(p[8:])

        K, _, _ = ct.lqr(A, B, Q, R)

        return np.asarray(K, dtype=float)

    if method == "PPC":
        desired = desired_ppc_poles(p)

        # scipy/control may emit many convergence warnings during PSO.
        # They are captured and counted instead of flooding the console.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            K = ct.place(A, B, desired)

        if diagnostics is not None and len(caught) > 0:
            diagnostics["ppc_place_warning_candidates"] += 1

        K = np.real_if_close(
            np.asarray(K)
        ).astype(float)

        actual = closed_loop_eigenvalues(K)

        pole_error = normalized_pole_matching_error(
            actual,
            desired,
        )

        if diagnostics is not None:
            diagnostics["ppc_max_pole_error"] = max(
                diagnostics["ppc_max_pole_error"],
                pole_error,
            )

        if pole_error > PPC_POLE_TOL:
            if diagnostics is not None:
                diagnostics["ppc_rejected_pole_mismatch"] += 1

            raise PolePlacementMismatch(
                f"PPC normalized pole mismatch {pole_error:.3e} "
                f"exceeds tolerance {PPC_POLE_TOL:.3e}."
            )

        return K

    if method == "DIRECT":
        return p.reshape(2, 8)

    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 5. CLOSED-LOOP SIMULATION
# =============================================================================

def simulate_controller(K):
    """Simulate the active suspension using u = -Kx."""
    system = ct.ss(
        A - B @ K,
        G,
        np.eye(8),
        np.zeros((8, 2)),
    )

    response = ct.forced_response(
        system,
        T=T_SIM,
        U=W,
    )

    states = np.asarray(response.states)
    control = -K @ states

    total_cost, parts = calculate_cost(
        response.time,
        states,
        control,
        W,
        return_parts=True,
    )

    return {
        "J": total_cost,
        "parts": parts,
        "time": np.asarray(response.time),
        "x": states,
        "u": np.asarray(control),
        "max_real_eig": maximum_real_eigenvalue(K),
    }


# =============================================================================
# 6. PSO OBJECTIVE
# =============================================================================

def new_diagnostics():
    return {
        "ppc_place_warning_candidates": 0,
        "ppc_rejected_pole_mismatch": 0,
        "ppc_max_pole_error": 0.0,
        "unstable_candidates": 0,
        "numerical_failures": 0,
    }


def objective(particles, method, diagnostics):
    costs = np.full(
        particles.shape[0],
        PENALTY_COST,
        dtype=float,
    )

    for idx, particle in enumerate(particles):
        try:
            K = controller_from_position(
                particle,
                method,
                diagnostics=diagnostics,
            )

            if maximum_real_eigenvalue(K) >= STABILITY_MARGIN:
                diagnostics["unstable_candidates"] += 1
                continue

            result = simulate_controller(K)
            costs[idx] = result["J"]

        except PolePlacementMismatch:
            # Already counted in PPC diagnostics.
            continue

        except Exception:
            diagnostics["numerical_failures"] += 1
            continue

    return costs


# =============================================================================
# 7. ONE INDEPENDENT OPTIMIZATION RUN
# =============================================================================

def set_random_seed(seed):
    """
    PySwarms relies on NumPy's global random state; Python's random module is
    seeded as well so the experiment remains explicit and deterministic.
    """
    np.random.seed(seed)
    random.seed(seed)


def optimize_once(method, seed):
    set_random_seed(seed)

    cfg = PSO_CONFIG[method]
    lower, upper = BOUNDS[method]

    diagnostics = new_diagnostics()

    optimizer = ps.single.GlobalBestPSO(
        n_particles=cfg["particles"],
        dimensions=cfg["dimensions"],
        options=PSO_OPTIONS,
        bounds=(lower, upper),
    )

    start = time.perf_counter()

    best_cost, best_position = optimizer.optimize(
        lambda p: objective(
            p,
            method,
            diagnostics,
        ),
        iters=cfg["iters"],
        verbose=SHOW_PSO_PROGRESS,
    )

    runtime_s = time.perf_counter() - start

    # Final controller from exactly the optimizer's returned coordinates.
    K = controller_from_position(
        best_position,
        method,
        diagnostics=None,
    )

    result = simulate_controller(K)

    stable = (
        result["max_real_eig"]
        < STABILITY_MARGIN
    )

    successful = (
        np.isfinite(result["J"])
        and result["J"] < PENALTY_COST
        and stable
    )

    row = {
        "method": method,
        "seed": int(seed),
        "optimizer_best_cost": float(best_cost),
        "J": float(result["J"]),
        "comfort": result["parts"]["comfort"],
        "stability": result["parts"]["stability"],
        "control_effort": result["parts"]["control_effort"],
        "smoothness": result["parts"]["smoothness"],
        "raw_comfort": result["parts"]["raw_comfort"],
        "raw_stability": result["parts"]["raw_stability"],
        "raw_control_effort": result["parts"]["raw_control_effort"],
        "raw_smoothness": result["parts"]["raw_smoothness"],
        "max_real_eig": float(result["max_real_eig"]),
        "stable": bool(stable),
        "successful": bool(successful),
        "runtime_s": float(runtime_s),
        "best_position": json.dumps(
            np.asarray(best_position).tolist()
        ),
        "K": json.dumps(
            np.asarray(K).tolist()
        ),
        **diagnostics,
    }

    return row


# =============================================================================
# 8. PASSIVE BASELINE
# =============================================================================

def passive_baseline():
    system = ct.ss(
        A,
        G,
        np.eye(8),
        np.zeros((8, 2)),
    )

    response = ct.forced_response(
        system,
        T=T_SIM,
        U=W,
    )

    zero_control = np.zeros(
        (2, len(T_SIM))
    )

    total_cost, parts = calculate_cost(
        response.time,
        response.states,
        zero_control,
        W,
        return_parts=True,
    )

    return {
        "J": float(total_cost),
        **parts,
    }


# =============================================================================
# 9. RESUME-SAFE RESULTS
# =============================================================================

def load_runs():
    if RUNS_CSV.exists():
        return pd.read_csv(RUNS_CSV)

    return pd.DataFrame()


def save_run(row):
    """
    Save immediately after each completed optimization so an interrupted study
    can be resumed later.
    """
    if RUNS_CSV.exists():
        df = pd.read_csv(RUNS_CSV)

        if not df.empty:
            duplicate = (
                (df["method"] == row["method"])
                & (df["seed"] == row["seed"])
            )

            df = df.loc[~duplicate].copy()

        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True,
        )

    else:
        df = pd.DataFrame([row])

    df = (
        df
        .sort_values(["method", "seed"])
        .reset_index(drop=True)
    )

    df.to_csv(
        RUNS_CSV,
        index=False,
    )


# =============================================================================
# 10. STATISTICAL SUMMARY
# =============================================================================

SUMMARY_METRICS = [
    "J",
    "comfort",
    "stability",
    "control_effort",
    "smoothness",
    "max_real_eig",
    "runtime_s",
]


def successful_runs(df):
    """
    Robust handling because pandas may read Boolean columns as bool or strings.
    """
    if df.empty:
        return df.copy()

    success = df["successful"]

    if success.dtype == bool:
        mask = success
    else:
        mask = (
            success.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1"])
        )

    return df.loc[mask].copy()


def summarize_runs(df):
    rows = []

    for method in METHODS:
        group = df.loc[
            df["method"] == method
        ].copy()

        if group.empty:
            continue

        good = successful_runs(group)

        row = {
            "method": method,
            "n_total": int(len(group)),
            "n_success": int(len(good)),
            "success_rate": (
                len(good) / len(group)
                if len(group)
                else np.nan
            ),
        }

        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(
                good[metric],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)

            if len(values) == 0:
                stats = {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }

            else:
                stats = {
                    "mean": float(np.mean(values)),
                    "std": float(
                        np.std(values, ddof=1)
                        if len(values) > 1
                        else 0.0
                    ),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

            for stat_name, stat_value in stats.items():
                row[f"{metric}_{stat_name}"] = stat_value

        # Aggregate diagnostics.
        diagnostic_columns = [
            "ppc_place_warning_candidates",
            "ppc_rejected_pole_mismatch",
            "unstable_candidates",
            "numerical_failures",
        ]

        for column in diagnostic_columns:
            if column in group.columns:
                row[f"{column}_total"] = float(
                    pd.to_numeric(
                        group[column],
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .sum()
                )

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    return summary


def print_summary(summary):
    columns = [
        "method",
        "n_success",
        "success_rate",
        "J_mean",
        "J_std",
        "J_median",
        "J_min",
        "J_max",
    ]

    print("\n" + "=" * 92)
    print("STATISTICAL SUMMARY")
    print("=" * 92)

    if not summary.empty:
        print(
            summary[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.6g}",
            )
        )

    print("=" * 92)


# =============================================================================
# 11. BEST CONTROLLERS
# =============================================================================

def decode_json_array(text):
    return np.asarray(
        json.loads(text),
        dtype=float,
    )


def save_best_controllers(df):
    good = successful_runs(df)
    payload = {}

    for method in METHODS:
        group = good.loc[
            good["method"] == method
        ]

        if group.empty:
            continue

        best_idx = pd.to_numeric(
            group["J"],
            errors="coerce",
        ).idxmin()

        best = group.loc[best_idx]

        payload[f"K_{method}"] = decode_json_array(
            best["K"]
        )

        payload[f"position_{method}"] = decode_json_array(
            best["best_position"]
        )

        payload[f"seed_{method}"] = np.array(
            [int(best["seed"])]
        )

        payload[f"J_{method}"] = np.array(
            [float(best["J"])]
        )

    if payload:
        np.savez(
            BEST_NPZ,
            **payload,
        )



def serialize_complex(values):
    """Return complex values as JSON-friendly real/imaginary dictionaries."""
    return [
        {
            "real": float(np.real(value)),
            "imag": float(np.imag(value)),
        }
        for value in np.asarray(values, dtype=complex)
    ]


def save_best_controllers_json(df):
    """
    Save a human-readable audit record for the best run of each method.

    The CSV already stores K and the optimized coordinates for every run.
    This JSON makes the synthesis parameters explicit:
    - LQR: diagonal Q and R, K, and achieved closed-loop poles;
    - PPC: alpha/beta coordinates, requested poles, K, achieved poles, and
      the normalized pole-assignment mismatch;
    - Direct: optimized K and achieved closed-loop poles.
    """
    good = successful_runs(df)
    payload = {
        "description": (
            "Best successful controller for each parameterization. "
            "All reported costs are evaluated with the common suspension objective."
        ),
        "methods": {},
    }

    for method in METHODS:
        group = good.loc[good["method"] == method]
        if group.empty:
            continue

        best_idx = pd.to_numeric(
            group["J"], errors="coerce"
        ).idxmin()
        best = group.loc[best_idx]

        position = decode_json_array(best["best_position"])
        K = decode_json_array(best["K"])
        achieved_poles = closed_loop_eigenvalues(K)

        record = {
            "seed": int(best["seed"]),
            "J": float(best["J"]),
            "weighted_cost_components": {
                "comfort": float(best["comfort"]),
                "tire_deflection": float(best["stability"]),
                "control_effort": float(best["control_effort"]),
                "smoothness": float(best["smoothness"]),
            },
            "raw_cost_components": {
                "comfort": float(best["raw_comfort"]),
                "tire_deflection": float(best["raw_stability"]),
                "control_effort": float(best["raw_control_effort"]),
                "smoothness": float(best["raw_smoothness"]),
            },
            "K": np.asarray(K, dtype=float).tolist(),
            "achieved_closed_loop_poles": serialize_complex(achieved_poles),
            "max_real_closed_loop_pole": float(
                np.max(np.real(achieved_poles))
            ),
            "optimization_budget": {
                "particles": int(PSO_CONFIG[method]["particles"]),
                "iterations": int(PSO_CONFIG[method]["iters"]),
                "decision_variables": int(PSO_CONFIG[method]["dimensions"]),
                "nominal_objective_evaluations": int(
                    PSO_CONFIG[method]["particles"]
                    * PSO_CONFIG[method]["iters"]
                ),
            },
        }

        if method == "LQR":
            record["optimized_parameters"] = {
                "Q_diagonal": position[:8].tolist(),
                "R_diagonal": position[8:].tolist(),
            }

        elif method == "PPC":
            requested_poles = desired_ppc_poles(position)
            record["optimized_parameters"] = {
                "alpha_beta_coordinates": position.tolist(),
                "requested_poles": serialize_complex(requested_poles),
            }
            record["normalized_pole_matching_error"] = float(
                normalized_pole_matching_error(
                    achieved_poles, requested_poles
                )
            )

        elif method == "DIRECT":
            record["optimized_parameters"] = {
                "K_entries_row_major": position.tolist(),
            }

        payload["methods"][method] = record

    BEST_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# 12. FIGURES
# =============================================================================

def ordered_cost_data(df, methods):
    good = successful_runs(df)

    present_methods = [
        method
        for method in methods
        if method in set(good["method"])
    ]

    values = [
        pd.to_numeric(
            good.loc[
                good["method"] == method,
                "J",
            ],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
        for method in present_methods
    ]

    return present_methods, values


def add_individual_points(ax, values, seed=12345):
    """
    Overlay individual optimization outcomes with a small deterministic
    horizontal jitter. The jitter is visual only and has no statistical role.
    """
    rng = np.random.default_rng(seed)

    for position, y in enumerate(values, start=1):
        x = rng.normal(
            loc=position,
            scale=0.035,
            size=len(y),
        )

        ax.scatter(
            x,
            y,
            s=20,
            alpha=0.50,
            zorder=3,
        )


def plot_boxplot_all_methods(df):
    """
    Complete comparison retained for transparency and repository use.
    Because PPC is much worse in the present study, the Direct-LQR separation
    is visually compressed in this figure.
    """
    methods, values = ordered_cost_data(
        df,
        ("DIRECT", "LQR", "PPC"),
    )

    if not values:
        return

    label_map = {
        "DIRECT": "PSO-Direct",
        "LQR": "PSO-LQR",
        "PPC": "PSO-PPC",
    }

    labels = [
        label_map[m]
        for m in methods
    ]

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    ax.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
        showfliers=False,
        widths=0.55,
    )

    add_individual_points(
        ax,
        values,
        seed=12345,
    )

    ax.set_ylabel(r"Total cost $J$")
    ax.set_xlabel("")
    ax.set_title(
        "Total cost over independent PSO runs"
    )
    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        BOXPLOT_ALL_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        BOXPLOT_ALL_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_boxplot_direct_lqr(df):
    """
    Manuscript-focused figure.

    PPC is intentionally omitted here because its much larger cost expands the
    vertical scale and obscures the practically relevant separation between
    PSO-Direct and PSO-LQR. PPC remains reported in the complete statistics and
    in cost_boxplot_all_methods.
    """
    methods, values = ordered_cost_data(
        df,
        ("DIRECT", "LQR"),
    )

    if not values:
        return

    label_map = {
        "DIRECT": "PSO-Direct",
        "LQR": "PSO-LQR",
    }

    labels = [
        label_map[m]
        for m in methods
    ]

    fig, ax = plt.subplots(
        figsize=(6.5, 5.0)
    )

    ax.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
        showfliers=False,
        widths=0.55,
    )

    add_individual_points(
        ax,
        values,
        seed=12345,
    )

    ax.set_ylabel(r"Total cost $J$")
    ax.set_xlabel("")
    ax.set_title(
        "PSO-Direct and PSO-LQR over independent runs"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    all_values = np.concatenate(values)

    if len(all_values) > 0:
        span = (
            float(all_values.max())
            - float(all_values.min())
        )

        margin = (
            0.10 * span
            if span > 0
            else 0.1
        )

        ax.set_ylim(
            float(all_values.min()) - margin,
            float(all_values.max()) + margin,
        )

    fig.tight_layout()

    fig.savefig(
        BOXPLOT_FOCUSED_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        BOXPLOT_FOCUSED_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_cost_by_run(df):
    good = successful_runs(df)

    if good.empty:
        return

    fig, ax = plt.subplots(
        figsize=(9.0, 5.0)
    )

    label_map = {
        "DIRECT": "PSO-Direct",
        "LQR": "PSO-LQR",
        "PPC": "PSO-PPC",
    }

    for method in METHODS:
        group = (
            good.loc[
                good["method"] == method
            ]
            .sort_values("seed")
        )

        if group.empty:
            continue

        ax.plot(
            group["seed"],
            group["J"],
            marker="o",
            label=label_map[method],
        )

    ax.set_xlabel("Random seed")
    ax.set_ylabel(r"Total cost $J$")
    ax.set_title(
        "Optimization outcome for each independent run"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RUNPLOT_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        RUNPLOT_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_best_responses(df):
    good = successful_runs(df)

    if good.empty:
        return

    best_results = {}

    for method in METHODS:
        group = good.loc[
            good["method"] == method
        ]

        if group.empty:
            continue

        best_idx = pd.to_numeric(
            group["J"],
            errors="coerce",
        ).idxmin()

        best = group.loc[best_idx]

        K = decode_json_array(
            best["K"]
        )

        best_results[method] = simulate_controller(K)

    passive_system = ct.ss(
        A,
        G,
        np.eye(8),
        np.zeros((8, 2)),
    )

    passive_response = ct.forced_response(
        passive_system,
        T=T_SIM,
        U=W,
    )

    label_map = {
        "DIRECT": "PSO-Direct",
        "LQR": "PSO-LQR",
        "PPC": "PSO-PPC",
    }

    # Keep each controller on the same default Matplotlib color in every panel.
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        "PASSIVE": cycle[0],
        "DIRECT": cycle[1],
        "LQR": cycle[2],
        "PPC": cycle[3],
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
    )

    # Body vertical displacement.
    axes[0, 0].plot(
        T_SIM,
        passive_response.states[0],
        "--",
        label="Passive",
        color=color_map["PASSIVE"],
    )

    for method, result in best_results.items():
        axes[0, 0].plot(
            T_SIM,
            result["x"][0],
            label=label_map[method],
            color=color_map[method],
        )

    axes[0, 0].set_title(
        "Body vertical displacement"
    )
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Displacement (m)")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend()

    # Pitch angle.
    axes[0, 1].plot(
        T_SIM,
        passive_response.states[2],
        "--",
        label="Passive",
        color=color_map["PASSIVE"],
    )

    for method, result in best_results.items():
        axes[0, 1].plot(
            T_SIM,
            result["x"][2],
            label=label_map[method],
            color=color_map[method],
        )

    axes[0, 1].set_title("Pitch angle")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Angle (rad)")
    axes[0, 1].grid(True, alpha=0.25)

    # Front actuator.
    for method, result in best_results.items():
        axes[1, 0].plot(
            T_SIM,
            result["u"][1],
            label=label_map[method],
            color=color_map[method],
        )

    axes[1, 0].set_title(
        "Front control force"
    )
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Force (N)")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend()

    # Rear actuator.
    for method, result in best_results.items():
        axes[1, 1].plot(
            T_SIM,
            result["u"][0],
            label=label_map[method],
            color=color_map[method],
        )

    axes[1, 1].set_title(
        "Rear control force"
    )
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Force (N)")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        BEST_RESPONSE_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        BEST_RESPONSE_PDF,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# 13. EXPERIMENT METADATA
# =============================================================================

def package_version(module):
    return getattr(
        module,
        "__version__",
        "unknown",
    )


def save_experiment_config(passive):
    """
    Save enough metadata to audit the numerical experiment without inspecting
    the source code line by line.
    """
    config = {
        "experiment": {
            "n_runs": N_RUNS,
            "base_seed": BASE_SEED,
            "methods": list(METHODS),
            "resume": RESUME,
            "stability_margin": STABILITY_MARGIN,
            "penalty_cost": PENALTY_COST,
            "ppc_pole_tolerance": PPC_POLE_TOL,
        },
        "pso_options": PSO_OPTIONS,
        "pso_config": PSO_CONFIG,
        "evaluation_budget": {
            method: {
                "objective_evaluations_per_run": int(
                    cfg["particles"] * cfg["iters"]
                ),
                "evaluations_per_decision_variable": float(
                    cfg["particles"] * cfg["iters"] / cfg["dimensions"]
                ),
            }
            for method, cfg in PSO_CONFIG.items()
        },
        "bounds": {
            "LQR_lower": LOWER_LQR.tolist(),
            "LQR_upper": UPPER_LQR.tolist(),
            "PPC_lower": LOWER_PPC.tolist(),
            "PPC_upper": UPPER_PPC.tolist(),
            "DIRECT_lower": LOWER_DIRECT.tolist(),
            "DIRECT_upper": UPPER_DIRECT.tolist(),
        },
        "model": {
            "Lf_m": Lf,
            "Lr_m": Lr,
            "Km_N_per_m": Km,
            "Cm_Ns_per_m": Cm,
            "M_kg": M,
            "Iy_kg_m2": Iy,
            "Mf_kg": Mf,
            "Mr_kg": Mr,
            "Cmf_Ns_per_m": Cmf,
            "Cmr_Ns_per_m": Cmr,
            "Kmf_N_per_m": Kmf,
            "Kmr_N_per_m": Kmr,
        },
        "simulation": {
            "dt_s": DT,
            "duration_s": float(T_SIM[-1]),
            "vehicle_speed_m_per_s": VEHICLE_SPEED,
            "axle_delay_s": AXLE_DELAY,
            "bump_height_m": BUMP_HEIGHT,
            "bump_length_m": BUMP_LENGTH,
            "bump_front_start_s": BUMP_START_FRONT,
            "bump_duration_s": BUMP_DURATION,
        },
        "cost_weights": {
            "comfort": W_COMFORT,
            "stability": W_STABILITY,
            "control_effort": W_CONTROL,
            "smoothness": W_SMOOTHNESS,
        },
        "passive_baseline": passive,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": package_version(np),
            "pandas": package_version(pd),
            "matplotlib": package_version(matplotlib),
            "scipy": package_version(scipy),
            "control": package_version(ct),
            "pyswarms": package_version(ps),
        },
    }

    CONFIG_JSON.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# 14. RUN THE EXPERIMENT
# =============================================================================

def run_experiment():
    existing = load_runs()

    completed = set()

    if RESUME and not existing.empty:
        completed = {
            (
                str(row["method"]),
                int(row["seed"]),
            )
            for _, row in existing.iterrows()
        }

        print(
            f"Resume enabled: {len(completed)} completed "
            f"method/seed pairs found."
        )

    total_jobs = (
        len(METHODS) * N_RUNS
    )

    job_number = 0

    for run_idx in range(N_RUNS):
        seed = BASE_SEED + run_idx

        print("\n" + "-" * 92)
        print(
            f"INDEPENDENT RUN {run_idx + 1}/{N_RUNS} "
            f"(seed={seed})"
        )
        print("-" * 92)

        for method in METHODS:
            job_number += 1
            key = (method, seed)

            if (
                RESUME
                and key in completed
            ):
                print(
                    f"[{job_number:03d}/{total_jobs:03d}] "
                    f"{method:6s} seed={seed}: skipped"
                )
                continue

            print(
                f"\n[{job_number:03d}/{total_jobs:03d}] "
                f"{method:6s} seed={seed}: START"
            )

            try:
                row = optimize_once(
                    method,
                    seed,
                )

                save_run(row)

                status = (
                    "OK"
                    if row["successful"]
                    else "FAILED"
                )

                print(
                    f"[{method}] {status}"
                    f" | J={row['J']:.6g}"
                    f" | max Re(lambda)="
                    f"{row['max_real_eig']:.3e}"
                    f" | runtime={row['runtime_s']:.1f} s"
                )

                if method == "PPC":
                    print(
                        "    PPC diagnostics:"
                        f" warning candidates="
                        f"{row['ppc_place_warning_candidates']},"
                        f" rejected mismatch="
                        f"{row['ppc_rejected_pole_mismatch']},"
                        f" max pole error="
                        f"{row['ppc_max_pole_error']:.3e}"
                    )

            except Exception as exc:
                error_row = {
                    "method": method,
                    "seed": seed,
                    "optimizer_best_cost": np.nan,
                    "J": np.nan,
                    "comfort": np.nan,
                    "stability": np.nan,
                    "control_effort": np.nan,
                    "smoothness": np.nan,
                    "raw_comfort": np.nan,
                    "raw_stability": np.nan,
                    "raw_control_effort": np.nan,
                    "raw_smoothness": np.nan,
                    "max_real_eig": np.nan,
                    "stable": False,
                    "successful": False,
                    "runtime_s": np.nan,
                    "best_position": "[]",
                    "K": "[]",
                    "ppc_place_warning_candidates": np.nan,
                    "ppc_rejected_pole_mismatch": np.nan,
                    "ppc_max_pole_error": np.nan,
                    "unstable_candidates": np.nan,
                    "numerical_failures": np.nan,
                    "error": repr(exc),
                }

                save_run(error_row)

                print(
                    f"[{method}] ERROR: {exc!r}"
                )


# =============================================================================
# 15. MAIN
# =============================================================================


def parse_cli_args():
    """Command-line convenience options; they do not alter the algorithms."""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce or post-process the half-car PSO controller comparison."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run-optimization",
        action="store_true",
        help="Run the full PSO experiment before generating outputs.",
    )
    mode.add_argument(
        "--results-only",
        action="store_true",
        help=(
            "Do not rerun PSO; regenerate statistics, controller exports, "
            "and figures from pso_repetitions_results/runs.csv."
        ),
    )
    return parser.parse_args()


def main():
    print("=" * 92)
    print("HALF-CAR PSO CONTROLLER COMPARISON")
    print("=" * 92)

    print(f"Script directory : {SCRIPT_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"N_RUNS           : {N_RUNS}")
    print(
        f"Seeds            : "
        f"{BASE_SEED} ... {BASE_SEED + N_RUNS - 1}"
    )
    print(f"Methods          : {METHODS}")
    print(f"Run optimization : {RUN_OPTIMIZATION}")

    passive = passive_baseline()

    print(
        f"\nPassive baseline J = "
        f"{passive['J']:.6f}"
    )

    # Preserve the metadata of the published experiment when only post-processing
    # bundled results. A new environment/config record is written when a new
    # optimization campaign is actually run, or if no config file exists yet.
    if RUN_OPTIMIZATION or not CONFIG_JSON.exists():
        save_experiment_config(passive)

    if RUN_OPTIMIZATION:
        run_experiment()

    if not RUNS_CSV.exists():
        raise FileNotFoundError(
            f"\nNo results file was found at:\n{RUNS_CSV}\n\n"
            "Set RUN_OPTIMIZATION = True to generate the results, "
            "or copy an existing runs.csv into the output folder."
        )

    df = pd.read_csv(
        RUNS_CSV
    )

    summary = summarize_runs(df)
    print_summary(summary)

    save_best_controllers(df)
    save_best_controllers_json(df)

    # Complete comparison for transparency / repository.
    plot_boxplot_all_methods(df)

    # Focused figure intended for the manuscript.
    plot_boxplot_direct_lqr(df)

    plot_cost_by_run(df)
    plot_best_responses(df)

    print("\nGenerated files:")
    print(f"  {RUNS_CSV}")
    print(f"  {SUMMARY_CSV}")
    print(f"  {BEST_NPZ}")
    print(f"  {BEST_JSON}")
    print(f"  {CONFIG_JSON}")
    print(f"  {BOXPLOT_ALL_PNG}")
    print(f"  {BOXPLOT_FOCUSED_PNG}")
    print(f"  {RUNPLOT_PNG}")
    print(f"  {BEST_RESPONSE_PNG}")

    print("\nManuscript reporting recommendation:")
    print(
        "  Report mean ± sample standard deviation, "
        "median [Q1, Q3], range, and success rate."
    )
    print(
        "  Use cost_boxplot_direct_lqr.png in the main text "
        "to resolve the difference between the two competitive methods."
    )
    print(
        "  Retain PPC in the numerical table and in "
        "cost_boxplot_all_methods.png for complete reporting."
    )


if __name__ == "__main__":
    args = parse_cli_args()
    if args.run_optimization:
        RUN_OPTIMIZATION = True
    elif args.results_only:
        RUN_OPTIMIZATION = False
    main()