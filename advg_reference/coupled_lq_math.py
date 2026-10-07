from __future__ import annotations
import math
import numpy as np
BENCHMARK_VERSION="coupled_lq_v1"
STATE_DIM=4
ACTION_DIM=2
ACTION_BOUND=5.

def coupled_lq_matrices() -> dict[str, np.ndarray]:
    """Return independent copies of the fixed, coupled benchmark matrices.

    A has strictly negative definite symmetric part, so passive dynamics are
    stable, with slow damping rather than an unstable-reset confound. Both
    controls affect both oscillator pairs. Q and R are positive definite;
    noise_std multiplies identity diffusion in all four state coordinates.
    Resets are N(0, reset_covariance), independently of the process RNG.
    """
    return {
        "A": np.asarray([
            [-.12, 1., 0., .25],
            [-1., -.20, .35, 0.],
            [0., -.35, -.15, .9],
            [-.25, 0., -.9, -.18],
        ], dtype=np.float64),
        "B": np.asarray([
            [.3, .1], [1., .35], [.15, .4], [-.3, 1.],
        ], dtype=np.float64),
        "Q": np.asarray([
            [2., .25, .15, 0.],
            [.25, .7, 0., .1],
            [.15, 0., 1.5, -.2],
            [0., .1, -.2, .6],
        ], dtype=np.float64),
        "R": np.asarray([[.15, .025], [.025, .20]], dtype=np.float64),
        "reset_covariance": np.diag([1., .5, 1., .5]),
    }

def held_action_discretization(dt: float, noise_std: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return F,G,W for X_next=F X+G a+N(0,W) under a held action.

    The covariance uses the matrix exponential identity for the integral
    int_0^dt exp(A s) sigma^2 I exp(A.T s) ds. It is not sigma^2*dt*I
    except to first order in dt. No environment or random generator is used.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be nonnegative and finite")
    from scipy.linalg import expm

    matrices = coupled_lq_matrices()
    A, B = matrices["A"], matrices["B"]
    drift_block = np.zeros((STATE_DIM + ACTION_DIM, STATE_DIM + ACTION_DIM))
    drift_block[:STATE_DIM, :STATE_DIM] = A
    drift_block[:STATE_DIM, STATE_DIM:] = B
    transition = expm(dt * drift_block)
    F = transition[:STATE_DIM, :STATE_DIM].copy()
    G = transition[:STATE_DIM, STATE_DIM:].copy()
    if noise_std == 0.:
        return F, G, np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    noise_block = np.zeros((2 * STATE_DIM, 2 * STATE_DIM), dtype=np.float64)
    noise_block[:STATE_DIM, :STATE_DIM] = A
    noise_block[:STATE_DIM, STATE_DIM:] = noise_std ** 2 * np.eye(STATE_DIM)
    noise_block[STATE_DIM:, STATE_DIM:] = -A.T
    covariance_block = expm(dt * noise_block)
    W = covariance_block[:STATE_DIM, STATE_DIM:] @ F.T
    W = .5 * (W + W.T)
    return F, G, W
