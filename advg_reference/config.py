"""One shared neural recipe; task files describe only the environment."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


TASK_FIELDS = {'task', 'seed', 'dt', 'microsteps', 'horizon', 'noise_std'}


@dataclass(frozen=True)
class Recipe:
    task: str
    seed: int
    dt: float
    microsteps: int
    horizon: float
    noise_std: float
    discount_rate: float = 0.2231435513142097
    exploration_std: float = 0.3
    width: int = 128
    num_envs: int = 4
    buffer_duration: float = 400.0
    batch_size: int = 64
    training_duration: float = 2000.0
    warmup_duration: float = 40.0
    update_interval: float = 0.16
    window_duration: float = 0.2
    terminal_weight: float = 0.002
    actor_init_scale: float = 0.01
    max_grad_norm: float = 10.0
    critic_lr: float = 0.0003
    actor_lr: float = 0.0001
    actor_cap: float = 0.0005
    actor_cap_final: float = 0.003
    actor_hold_duration: float = 400.0
    cap_ramp_start: float = 400.0
    cap_ramp_end: float = 1200.0
    evaluation_interval: float = 400.0
    curve_episodes: int = 10
    curve_seed_base: int = 9780000
    final_episodes: int = 50
    final_seed_base: int = 9790000

    def __post_init__(self):
        self.validate()

    @property
    def micro_dt(self):
        return self.dt / self.microsteps

    @property
    def gamma(self):
        return math.exp(-self.discount_rate * self.dt)

    @property
    def episode_steps(self):
        return round(self.horizon / self.dt)

    @property
    def window_cells(self):
        return round(self.window_duration / self.dt)

    @property
    def buffer_size(self):
        return round(self.buffer_duration / self.dt)

    @property
    def total_timesteps(self):
        return round(self.training_duration / self.dt)

    @property
    def warmup_steps(self):
        return round(self.warmup_duration / self.dt)

    @property
    def train_freq(self):
        return round(self.update_interval / (self.num_envs * self.dt))

    @property
    def eval_every(self):
        return round(self.evaluation_interval / self.dt)

    def validate(self):
        integers = ('width', 'num_envs', 'batch_size', 'microsteps', 'curve_episodes', 'final_episodes')
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('seed', 'curve_seed_base', 'final_seed_base'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        nonnegative = ('noise_std', 'terminal_weight', 'actor_init_scale', 'actor_hold_duration', 'cap_ramp_start')
        for field in fields(self):
            name = field.name
            if name == 'task' or name in integers or name in ('seed', 'curve_seed_base', 'final_seed_base'):
                continue
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and name not in nonnegative):
                raise ValueError(f'{name} must be finite and positive (nonnegative for noise/penalties)')
        if not isinstance(self.task, str) or not self.task:
            raise ValueError('task must be named')
        ratios = [(self.horizon/self.dt), (self.window_duration/self.dt)]
        ratios += [getattr(self, name)/(self.num_envs*self.dt) for name in
                   ('buffer_duration', 'training_duration', 'warmup_duration', 'update_interval', 'evaluation_interval')]
        if any(x < 1 or not math.isclose(x, round(x), rel_tol=0, abs_tol=1e-8) for x in ratios):
            raise ValueError('durations must contain whole cells/vector steps; update_interval must be at least one vector step')
        if self.actor_cap_final < self.actor_cap or self.cap_ramp_end <= self.cap_ramp_start:
            raise ValueError('actor-cap ramp must be increasing')
        if self.window_cells > self.episode_steps or self.buffer_size//self.num_envs < self.window_cells:
            raise ValueError('episode and replay must contain the complete window')
        if self.warmup_steps < self.num_envs*self.window_cells or self.warmup_steps >= self.total_timesteps:
            raise ValueError('warmup must acquire a complete window per environment and leave a training budget')


def load_recipe(task_path, algorithm_path):
    task = json.loads(Path(task_path).read_text(encoding='utf-8'))
    algorithm = json.loads(Path(algorithm_path).read_text(encoding='utf-8'))
    if set(task) != TASK_FIELDS:
        raise ValueError('task file must contain only task, seed, dt, microsteps, horizon and noise_std')
    expected = {field.name for field in fields(Recipe)} - TASK_FIELDS
    if set(algorithm) != expected:
        raise ValueError('algorithm file must contain the complete shared settings; task overrides are not allowed')
    return Recipe(**task, **algorithm)
