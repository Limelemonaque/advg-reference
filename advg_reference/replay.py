"""Serializable vector replay with fixed CT windows and terminal-shortened n-step draws.

Rewards are already integrated over one physical cell. The collector supplies
the actual terminal successor, never the auto-reset observation. All sampling
uses this object's private NumPy generator. No simulator object is retained.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


class ReplayNotReady(RuntimeError):
    """No complete requested window has been acquired yet."""


class VectorReplay:
    FORMAT_VERSION = 1

    def __init__(self, cfg, observation_space, action_space, seed=None):
        self.num_envs = int(cfg.num_envs)
        self.observation_dim = int(observation_space.shape[0])
        self.action_dim = int(action_space.shape[0])
        self.capacity_steps = int(cfg.buffer_size) // self.num_envs
        if self.capacity_steps < 1:
            raise ValueError("Replay must hold a complete vector step")
        if len(observation_space.shape) != 1 or len(action_space.shape) != 1:
            raise ValueError("Replay requires flat observation and action spaces")
        self.capacity = self.capacity_steps * self.num_envs
        base = (self.capacity_steps, self.num_envs)
        self.observations = np.zeros((*base, self.observation_dim), np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros((*base, self.action_dim), np.float32)
        self.rewards = np.zeros(base, np.float32)
        self.dones = np.zeros(base, np.bool_)
        self.episode_ids = np.zeros(base, np.int64)
        self.current_episode_ids = np.zeros(self.num_envs, np.int64)
        self.total_vector_steps = 0
        self.filled_vector_steps = 0
        self.rng = np.random.default_rng(np.random.SeedSequence([
            int(cfg.seed if seed is None else seed), 81017]))

    def __len__(self):
        return self.filled_vector_steps * self.num_envs

    @property
    def size(self):
        return len(self)

    @property
    def position(self):
        return self.total_vector_steps % self.capacity_steps

    @property
    def total_transitions(self):
        return self.total_vector_steps * self.num_envs

    @property
    def allocated_bytes(self):
        return sum(getattr(self, key).nbytes for key in self._array_names())

    @staticmethod
    def _array_names():
        return ("observations", "next_observations", "actions", "rewards",
                "dones", "episode_ids")

    def add(self, obs, actions, rewards, next_obs, dones):
        values = (np.asarray(obs, dtype=np.float32),
                  np.asarray(actions, dtype=np.float32),
                  np.asarray(rewards, dtype=np.float32).reshape(-1),
                  np.asarray(next_obs, dtype=np.float32),
                  np.asarray(dones).reshape(-1))
        shapes = ((self.num_envs, self.observation_dim),
                  (self.num_envs, self.action_dim), (self.num_envs,),
                  (self.num_envs, self.observation_dim), (self.num_envs,))
        for value, shape in zip(values, shapes):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Replay received invalid shape/value: expected {shape}")
        obs, actions, rewards, next_obs, dones = values
        if not np.isin(dones, (0, 1)).all():
            raise ValueError("Terminal flags must be boolean")
        if self.filled_vector_steps:
            previous = (self.total_vector_steps - 1) % self.capacity_steps
            continuing = ~self.dones[previous]
            if not np.array_equal(self.next_observations[previous, continuing],
                                  obs[continuing]):
                raise ValueError("Nonterminal replay stream is not contiguous")
        slot = self.position
        self.observations[slot] = obs
        self.actions[slot] = actions
        self.rewards[slot] = rewards
        self.next_observations[slot] = next_obs
        self.dones[slot] = dones
        self.episode_ids[slot] = self.current_episode_ids
        self.current_episode_ids += dones.astype(np.int64)
        self.total_vector_steps += 1
        self.filled_vector_steps = min(self.filled_vector_steps + 1, self.capacity_steps)

    def _draw_cells(self, count):
        if count < 1 or not self.filled_vector_steps:
            raise ReplayNotReady("Replay is empty")
        oldest = self.total_vector_steps - self.filled_vector_steps
        times = self.rng.integers(oldest, self.total_vector_steps, size=count)
        envs = self.rng.integers(self.num_envs, size=count)
        return times, envs

    def sample_transitions(self, batch_size):
        times, envs = self._draw_cells(int(batch_size))
        slots = times % self.capacity_steps
        return {"observations": self.observations[slots, envs].copy(),
                "next_observations": self.next_observations[slots, envs].copy(),
                "actions": self.actions[slots, envs].copy(),
                "rewards": self.rewards[slots, envs].copy(),
                "dones": self.dones[slots, envs].copy()}

    def sample_states(self, batch_size):
        times, envs = self._draw_cells(int(batch_size))
        return self.observations[times % self.capacity_steps, envs].copy()

    def _lengths(self, starts, envs, length, allow_terminal_prefix):
        offsets = np.arange(length)
        times = starts[:, None] + offsets
        available = times < self.total_vector_steps
        terminal = self.dones[times % self.capacity_steps, envs[:, None]] & available
        first_terminal = np.where(terminal.any(1), terminal.argmax(1) + 1, length)
        if allow_terminal_prefix:
            lengths = first_terminal
            valid = (starts + lengths <= self.total_vector_steps)
        else:
            lengths = np.full(len(starts), length, dtype=np.int64)
            valid = (starts + length <= self.total_vector_steps)
            valid &= ~terminal[:, :-1].any(1)
        return valid, lengths

    def sample_windows(self, batch_size, cells, *, allow_terminal_prefix=False):
        """Uniform eligible start draws, never crossing reset or capacity wrap.

        CT uses exactly ``cells``. n-step allows any shorter prefix that ends at
        a real terminal; nonterminal incomplete prefixes are never bootstrapped
        from unacquired data. There is no silent one-cell fallback.
        """
        batch_size, length = int(batch_size), int(cells)
        if batch_size < 1 or length < 1 or length > self.capacity_steps:
            raise ValueError("Invalid replay batch/window size")
        if not self.filled_vector_steps:
            raise ReplayNotReady("Replay is empty")
        starts_out, envs_out, lengths_out = [], [], []
        remaining = batch_size
        enumeration = False
        for _ in range(4):
            starts, envs = self._draw_cells(max(32, 2 * remaining))
            valid, lengths = self._lengths(starts, envs, length, allow_terminal_prefix)
            keep = np.flatnonzero(valid)[:remaining]
            starts_out.append(starts[keep])
            envs_out.append(envs[keep])
            lengths_out.append(lengths[keep])
            remaining -= len(keep)
            if not remaining:
                break
        if remaining:
            enumeration = True
            oldest = self.total_vector_steps - self.filled_vector_steps
            eligible = []
            for first in range(0, len(self), 8192):
                ids = np.arange(first, min(first + 8192, len(self)))
                starts = oldest + ids // self.num_envs
                envs = ids % self.num_envs
                valid, lengths = self._lengths(starts, envs, length, allow_terminal_prefix)
                eligible.append(np.stack((starts[valid], envs[valid], lengths[valid]), axis=1))
            eligible = np.concatenate(eligible, axis=0)
            if not len(eligible):
                raise ReplayNotReady("No eligible fixed window has been acquired")
            selected = eligible[self.rng.integers(len(eligible), size=remaining)]
            starts_out.append(selected[:, 0])
            envs_out.append(selected[:, 1])
            lengths_out.append(selected[:, 2])
        starts = np.concatenate(starts_out)
        envs = np.concatenate(envs_out)
        lengths = np.concatenate(lengths_out)
        offsets = np.arange(length)[None, :]
        valid = offsets < lengths[:, None]
        # Repeat the terminal cell in padding; consumers must use valid_mask.
        safe_offsets = np.minimum(offsets, lengths[:, None] - 1)
        slots = (starts[:, None] + safe_offsets) % self.capacity_steps
        observations = self.observations[slots, envs[:, None]]
        successors = self.next_observations[slots, envs[:, None]]
        return {"states": np.concatenate((observations[:, :1], successors), axis=1),
                "actions": self.actions[slots, envs[:, None]].copy(),
                "rewards": self.rewards[slots, envs[:, None]].copy(),
                "dones": self.dones[slots, envs[:, None]].copy(),
                "lengths": lengths, "valid_mask": valid,
                "window_cells": length, "requested_window_cells": length,
                "length_fallback": 0, "enumeration_fallback": int(enumeration),
                **self._window_extras(slots, envs[:, None], starts[:, None] + safe_offsets)}

    def _window_extras(self, slots, envs, times):
        """Optional collection metadata; must not draw randomness or alter sampling."""
        return {}

    def state_dict(self):
        filled = self.filled_vector_steps
        return {"format_version": self.FORMAT_VERSION, "capacity_steps": self.capacity_steps,
                "num_envs": self.num_envs, "observation_dim": self.observation_dim,
                "action_dim": self.action_dim, "total_vector_steps": self.total_vector_steps,
                "filled_vector_steps": filled, "position": self.position,
                "current_episode_ids": self.current_episode_ids.copy(),
                "arrays": {key: getattr(self, key)[:filled].copy() for key in self._array_names()},
                "rng_state": deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        for key in ("format_version", "capacity_steps", "num_envs", "observation_dim", "action_dim"):
            expected = self.FORMAT_VERSION if key == "format_version" else getattr(self, key)
            if state[key] != expected:
                raise ValueError(f"Replay checkpoint mismatch: {key}")
        total, filled = int(state["total_vector_steps"]), int(state["filled_vector_steps"])
        if total < 0 or filled != min(total, self.capacity_steps):
            raise ValueError("Invalid replay cursor")
        if int(state["position"]) != total % self.capacity_steps:
            raise ValueError("Invalid replay position")
        for key in self._array_names():
            target = getattr(self, key)
            source = np.asarray(state["arrays"][key])
            if source.shape != target[:filled].shape or source.dtype != target.dtype:
                raise ValueError(f"Replay checkpoint array mismatch: {key}")
            if not np.isfinite(source).all():
                raise ValueError(f"Nonfinite replay checkpoint array: {key}")
            target.fill(0)
            target[:filled] = source
        episode_ids = np.asarray(state["current_episode_ids"], dtype=np.int64)
        if episode_ids.shape != (self.num_envs,):
            raise ValueError("Invalid episode counters")
        self.current_episode_ids[:] = episode_ids
        self.total_vector_steps, self.filled_vector_steps = total, filled
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
