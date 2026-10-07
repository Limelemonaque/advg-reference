from __future__ import annotations
from typing import Any
import math
import numpy as np
import gymnasium as gym

def _check_seed(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

def _array(value: Any, shape: tuple[int, ...], name: str,
           dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result.copy()

def euler_maruyama_step(state: np.ndarray, torque: float, micro_dt: float,
                        noise_std: float, standard_normal: float) -> np.ndarray:
    """One simultaneous Euler--Maruyama increment; no RNG is hidden here."""
    theta, omega = state
    drift = 15. * math.sin(float(theta)) + 3. * torque - .25 * omega
    return np.asarray([theta + micro_dt * omega,
                       omega + micro_dt * drift
                       + noise_std * math.sqrt(micro_dt) * standard_normal],
                      dtype=np.float64)

class StochasticPendulumEnv(gym.Env):
    """Raw SDE cell simulator; FiniteHorizonTask owns the finite horizon."""

    metadata = {"render_modes": []}
    reward_is_integrated = True

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.dt = config.dt
        self.micro_dt = config.micro_dt
        self.microsteps = int(round(self.dt / self.micro_dt))
        self.action_space = gym.spaces.Box(-2., 2., shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.asarray([-1., -1., -np.inf], dtype=np.float32),
            high=np.asarray([1., 1., np.inf], dtype=np.float32), dtype=np.float32)
        self.state: np.ndarray | None = None
        self.last_u: float | None = None
        self._process_rng: np.random.Generator | None = None

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("reset the SDE before requesting an observation")
        theta, omega = self.state
        return np.asarray([math.cos(float(theta)), math.sin(float(theta)), omega],
                          dtype=np.float32)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._process_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
        elif self._process_rng is None:
            process_seed = int(self.np_random.integers(0, 2**63))
            self._process_rng = np.random.default_rng(process_seed)
        if options is not None and set(options) - {"state"}:
            raise ValueError("the SDE reset accepts only the optional state field")
        if options is not None and "state" in options:
            self.state = _array(options["state"], (2,), "reset state")
        else:
            self.state = self.np_random.uniform(low=[-math.pi, -1.], high=[math.pi, 1.])
        self.last_u = None
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None or self._process_rng is None:
            raise RuntimeError("reset the SDE before stepping")
        command = _array(action, (1,), "action")
        torque = float(np.clip(command[0], -2., 2.))
        reward = 0.
        for _ in range(self.microsteps):
            theta, omega = self.state
            angle = (theta + math.pi) % (2. * math.pi) - math.pi
            reward_rate = -(angle * angle + .1 * omega * omega + .001 * torque * torque)
            reward += self.micro_dt * float(reward_rate)
            normal = float(self._process_rng.standard_normal())
            self.state = euler_maruyama_step(self.state, torque, self.micro_dt,
                                             self.config.noise_std, normal)
        if not np.isfinite(self.state).all() or not math.isfinite(reward):
            raise FloatingPointError("the SDE produced a nonfinite state or reward")
        self.last_u = torque
        return self._get_obs(), reward, False, False, {
            "reward_rate": reward / self.dt,
            "simulator_microsteps": self.microsteps,
            "process_diffusion_std": self.config.noise_std,
        }

class CoupledLQEnv(gym.Env):
    """Exact linear Gaussian cell; the shared wrapper owns the finite horizon.

    Fixed matrices and model-aware reference helpers live in coupled_lq_math.
    Ordinary agents receive only the common observation/step interface.
    """

    metadata = {"render_modes": []}
    reward_is_integrated = True

    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.task != "CoupledLQ-v1":
            raise ValueError("CoupledLQEnv requires CoupledLQ-v1")
        config.validate()
        try:
            from .coupled_lq_math import (ACTION_BOUND, BENCHMARK_VERSION,
                                          coupled_lq_matrices, held_action_discretization)
        except ImportError:
            from coupled_lq_math import (ACTION_BOUND, BENCHMARK_VERSION,
                                         coupled_lq_matrices, held_action_discretization)
        self.config = config
        self.dt = config.dt
        self.benchmark_version = BENCHMARK_VERSION
        matrices = coupled_lq_matrices()
        self.Q, self.R = matrices["Q"], matrices["R"]
        self.reset_covariance = matrices["reset_covariance"]
        self.F, self.G, self.process_covariance = held_action_discretization(self.dt, config.noise_std)
        self._reset_factor = np.linalg.cholesky(self.reset_covariance)
        self._process_factor = (np.linalg.cholesky(self.process_covariance)
                                if config.noise_std > 0. else np.zeros((4, 4)))
        self.action_space = gym.spaces.Box(-ACTION_BOUND, ACTION_BOUND, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.state: np.ndarray | None = None
        self.last_action: np.ndarray | None = None
        self._process_rng: np.random.Generator | None = None

    def _get_obs(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("reset CoupledLQ before requesting an observation")
        return self.state.astype(np.float32)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._process_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(31415,)))
        elif self._process_rng is None:
            self._process_rng = np.random.default_rng(int(self.np_random.integers(0, 2**63)))
        if options is not None and set(options) - {"state"}:
            raise ValueError("CoupledLQ reset accepts only the optional state field")
        self.state = (_array(options["state"], (4,), "reset state")
                      if options is not None and "state" in options
                      else self._reset_factor @ self.np_random.standard_normal(4))
        self.last_action = None
        return self._get_obs(), {"benchmark_version": self.benchmark_version}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.state is None or self._process_rng is None:
            raise RuntimeError("reset CoupledLQ before stepping")
        command = _array(action, (2,), "action")
        command = np.clip(command, self.action_space.low, self.action_space.high)
        # The left endpoint and executed action determine the whole cell cost.
        # Do not silently replace this quadrature by an integrated LQ reward.
        reward = -.5 * self.dt * float(self.state @ self.Q @ self.state + command @ self.R @ command)
        following = self.F @ self.state + self.G @ command
        if self.config.noise_std > 0.:
            following += self._process_factor @ self._process_rng.standard_normal(4)
        if not np.isfinite(following).all() or not math.isfinite(reward):
            raise FloatingPointError("CoupledLQ produced a nonfinite state or reward")
        self.state = following
        self.last_action = command.copy()
        return self._get_obs(), reward, False, False, {
            "reward_rate": reward / self.dt,
            "reward_quadrature": "left_endpoint",
            "simulator_microsteps": 1,
            "process_diffusion_std": self.config.noise_std,
            "process_noise_scheme": "exact_held_action_gaussian",
            "benchmark_version": self.benchmark_version,
        }
