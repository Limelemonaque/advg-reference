"""The paper's LQ and stochastic Pendulum environments, with explicit terminal storage."""
import math
import numpy as np
import gymnasium as gym

from .environments import CoupledLQEnv, StochasticPendulumEnv


class FiniteHorizonTask:
    def __init__(self, cfg, seed):
        if cfg.task not in ("CoupledLQ-v1", "StochasticPendulum-v1"):
            raise ValueError("the included examples support coupled LQ and stochastic Pendulum")
        self.cfg = cfg
        self.native = CoupledLQEnv(cfg) if cfg.task == "CoupledLQ-v1" else StochasticPendulumEnv(cfg)
        self.action_space = self.native.action_space
        low = np.concatenate((self.native.observation_space.low, [0., -1., -1.])).astype(np.float32)
        high = np.concatenate((self.native.observation_space.high, [1., 1., 1.])).astype(np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.seed = seed
        self.initialized = False
        self.finished = True
        self.elapsed_steps = 0

    def _observation(self, observation):
        fraction = self.elapsed_steps / self.cfg.episode_steps
        clock = ([1., 1., 0.] if self.elapsed_steps == self.cfg.episode_steps else
                 [fraction, math.cos(2 * math.pi * fraction), math.sin(2 * math.pi * fraction)])
        return np.concatenate((np.asarray(observation, np.float32), np.asarray(clock, np.float32)))

    def reset(self, *, seed=None):
        if seed is None and not self.initialized:
            seed = self.seed
        observation, _ = self.native.reset(seed=seed)
        self.elapsed_steps = 0
        self.initialized, self.finished = True, False
        return self._observation(observation)

    def step(self, action):
        if not self.initialized or self.finished:
            raise RuntimeError("reset before stepping an uninitialized or finished episode")
        command = np.asarray(action, dtype=self.action_space.dtype)
        command = np.clip(command, self.action_space.low, self.action_space.high)
        observation, reward, native_terminal, truncated, _ = self.native.step(command)
        if truncated:
            raise RuntimeError("unexpected native truncation")
        self.elapsed_steps += 1
        self.finished = bool(native_terminal or self.elapsed_steps >= self.cfg.episode_steps)
        return self._observation(observation), float(reward), self.finished

    def close(self):
        self.native.close()


class ParallelTasks:
    """Serial vector collector matching float32 paper replay and auto-reset behavior."""
    def __init__(self, cfg):
        seeds = [int(np.random.SeedSequence([cfg.seed, 51073, i]).generate_state(1, dtype=np.uint64)[0])
                 for i in range(cfg.num_envs)]
        self.envs = [FiniteHorizonTask(cfg, seed) for seed in seeds]
        self.observation_space, self.action_space = self.envs[0].observation_space, self.envs[0].action_space

    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions):
        following, replay_following, rewards, dones = [], [], [], []
        for env, action in zip(self.envs, actions):
            observation, reward, done = env.step(action)
            replay_following.append(observation.copy())
            following.append(env.reset() if done else observation)
            rewards.append(reward)
            dones.append(done)
        return np.stack(following), np.asarray(rewards, np.float32), np.asarray(dones, bool), np.stack(replay_following)

    def close(self):
        for env in self.envs:
            env.close()
