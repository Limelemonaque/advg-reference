# AdvG: a neural actor–critic reference

**Version 0.4.0.** This package provides one AdvG recipe with fully trainable
neural networks. The LQ and stochastic Pendulum examples use the same networks,
critic updates, optimizers and training schedule. Start by running an example;
then connect the same learner to your own simulator.

## What AdvG does

An actor chooses a continuous action. A value network estimates the return from
the current state, and an advantage network learns how changing the action
changes that return. The actor improves by following the advantage network's
action derivative, as in DDPG.

AdvG's distinguishing step is how it trains the advantage network. Divide a
trajectory into observation intervals, called **cells**. Each cell contains the
state, the action actually executed, the reward earned during that interval and
the next state. AdvG pairs the action dependence in each cell with the reward
and value change from that same cell, then adds these contributions over a
short window. [ALGORITHM.md](ALGORITHM.md) gives the exact losses and gradient
stops for implementing this update independently.

## 1. Install and check the package

Use Python **3.10–3.12**. Open a terminal in the directory containing this README
and `pyproject.toml`, then create a virtual environment:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on macOS/Linux. Install the tested dependencies and
package:

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

The recorded environment is Windows, Python 3.12.10 and CPU PyTorch 2.5.1.
A GPU, external dataset and the parent research repository are unnecessary.

## 2. Run an included example

First check that collection, critic updates and evaluation execute:

```text
python -m examples.train --config configs/lq.json --smoke --output runs/lq-smoke
python -m examples.train --config configs/pendulum.json --smoke --output runs/pendulum-smoke
```

A smoke run finishes before the scheduled actor start, so **zero actor updates
is expected**. It checks execution; use the full runs below to test learning:

```text
python -m examples.train --config configs/lq.json --output runs/lq
python -m examples.train --config configs/pendulum.json --output runs/pendulum
```

Each full run acquires 100,000 training transitions at `dt=0.02`. Use a new
output directory for every run; the runner refuses to overwrite an existing
one. Add `--seed 20801` to change the training seed. Add `--dt 0.005` to collect
on a finer time mesh with the same physical durations; this increases the
training budget to 400,000 transitions.

During training, the terminal prints the transition count and evaluation
return. Higher return is better. The output directory contains:

| File | What to read or use |
|---|---|
| `result.json` | Completion status, update counts and final mean return |
| `final_evaluation.json` | Final policy's returns on 50 evaluation episodes |
| `evaluations.jsonl` | Evaluation returns during training |
| `diagnostics.jsonl` | Critic residuals, gradient norms and actor movement |
| `manifest.json`, `source_snapshot/` | Resolved settings, dependency versions and exact training source |
| `parameters.pt` | Final network weights for inference; optimizer/replay state is not saved for resuming training |

Evaluation calls `agent.act(observations)` without exploration noise and uses
separate simulator seeds. The reported final result uses the final policy.

## 3. Understand the one shared recipe

All training settings live in **[configs/algorithm.json](configs/algorithm.json)**.
The two task files describe only the environment name, seed, observation time
step, simulator substeps, horizon and process noise. Observation dimensions and
action bounds come from the environment.

Every call to `agent.update()` performs the following steps:

1. Sample a minibatch of consecutive replay windows. For each cell, compare
   reward plus next-state value with the value at its start, using the current
   value network. Freeze these responses and the multi-step value target before
   changing any network parameters.
2. Update the value network **once against the frozen multi-step target**.
3. Update the advantage network once, pairing each cell's action dependence
   with its own frozen response.
4. Once the actor start time is reached, update the actor using the **newly
   updated advantage network**, with a cap on the resulting action movement.

The advantage is represented by `q(s,a) = F(s,a) - F(s,mu(s))`, where `mu` is the
actor and `F` is a learned state–action network. This makes the advantage zero
at the current actor's action. During critic fitting the actor's action is held
fixed, and both evaluations of `F` receive critic gradients. There is no
separate target-value network. Each of the actor, value and state–action
networks has two trainable ReLU hidden layers of width 128.

The following values are shared across both examples:

| Setting | Default |
|---|---|
| Critic / actor learning rate | `3e-4` / `1e-4`, using Adam |
| Exploration | Gaussian action noise, standard deviation `0.3`, clipped to action bounds |
| Minibatch | 64 windows; one length drawn uniformly from 1 to `round(0.2/dt)` per minibatch |
| Discount per cell | `exp(-0.2231435513142097 * dt)` |
| Collection warmup | 2,000 transitions at `dt=0.02` |
| Critic update interval | Every 8 acquired transitions at `dt=0.02` |
| Actor start | After acquiring 20,000 transitions at `dt=0.02` |
| Actor movement cap | Normalized RMS action change increases from `0.0005` to `0.003` |

Collection uses four environments. Acquisition durations in the configuration
sum time over those environments; a window's duration and the episode horizon
refer to one trajectory. The configuration converts durations to transition
counts automatically when `dt` changes. See [ALGORITHM.md](ALGORITHM.md) for
the complete schedule and formulas.

## 4. Connect another simulator

For a new task, define the state observation, bounded action and reward.
Include any time or conditioning information needed to determine future
dynamics, then provide these through the simulator adapter below.

The following is a complete collection and training loop using an included
environment. Run it from this README's directory after installation. For your
project, replace `ParallelTasks` with your simulator adapter; keep the learner
and update schedule for the first pilot. The command-line runner above adds
evaluation, logging and saved parameters to this loop.

```python
import torch

from advg_reference import AdvG, load_recipe
from advg_reference.tasks import ParallelTasks

cfg = load_recipe("configs/lq.json", "configs/algorithm.json")
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(cfg.seed)

tasks = ParallelTasks(cfg)
agent = AdvG(
    tasks.observation_space.shape[0],
    tasks.action_space.low,
    tasks.action_space.high,
    cfg,
)
obs = tasks.reset()

try:
    for acquired in range(cfg.num_envs, cfg.total_timesteps + 1, cfg.num_envs):
        actions = agent.act(obs, explore=True)
        following, rewards, dones, actual_next = tasks.step(actions)
        agent.replay.add(obs, actions, rewards, actual_next, dones)
        obs = following

        interval = cfg.num_envs * cfg.train_freq
        if (
            acquired > cfg.warmup_steps
            and (acquired - cfg.warmup_steps) % interval == 0
            and agent.replay.filled_vector_steps >= cfg.window_cells
        ):
            metrics = agent.update()
finally:
    tasks.close()
```

Your adapter must provide `reset()`, `step(actions)`, `close()`, an observation
space with `shape=(S,)`, and finite action bounds `low`/`high` of shape `(A,)`.
Here `N=cfg.num_envs`, `S` is the observation dimension and `A` is the action
dimension. The data returned by `step` must have these meanings:

| Array | Shape | Meaning |
|---|---|---|
| `actions` | `(N, A)` | The actions returned by the agent and actually executed |
| `rewards` | `(N,)` | Reward integrated over each observation interval |
| `dones` | `(N,)` | Boolean flags indicating true episode endings |
| `actual_next` | `(N, S)` | The actual successor, including the terminal observation before any reset |
| `following` | `(N, S)` | The next observation for collection, already reset where an episode ended |

Use flat `float32` observations, actions and rewards. Store consecutive
transitions from each environment; replay selects windows without crossing
resets. Each stored transition must span the configured constant interval `dt`.
Convert a reward rate to an interval reward exactly once. An arbitrary
collection cutoff is not a true terminal and must retain value bootstrap.

**Time and terminal value must match the supplied learner.** Append
`[u, cos(2*pi*u), sin(2*pi*u)]` as the last three observation coordinates, where
`u` increases from 0 to 1 over the episode. Use exactly `[1, 1, 0]` at a true
terminal. The learner uses these coordinates to enforce zero value remaining
after termination. A terminal payoff can be included in the final transition's
reward, with the discount convention accounted for. If you instead require a
nonzero terminal value or a different clock encoding, update the corresponding
boundary construction in `AdvG.value_loss` as well as the adapter.

The included `ParallelTasks` recognizes only LQ and Pendulum. A new task needs
its own adapter; changing the task name in JSON alone does not implement it.
Copy a task file for your environment's metadata and keep the shared algorithm
file for the initial pilot. [ADAPTING.md](ADAPTING.md) explains custom neural
architectures, structured observations and planning an initial pilot.

## Optional refinements after the first pilot

**Keep the one-value-update recipe above as the starting point.** We have also
tried the following implementation choices in earlier experiments. They may be
useful when adapting AdvG, depending on the task. This section describes later
experiments to consider; the supplied learner has no switches that enable
these changes automatically.

| Choice | What changes and why it may help | What we have checked |
|---|---|---|
| **More value fitting per update** | Take several value-optimizer steps on the same sampled batch and frozen target. This gives the value network more fitting work before the next training call. | Four passes improved the mean LQ result but reduced the mean Pendulum result relative to one pass in the [matched comparison](VERIFICATION.md#matched-training-comparison). |
| **A slowly updated value target** | Keep a second value network whose weights follow a moving average of the learned value network. Use that copy for bootstrap targets to reduce rapid changes in the training labels. | Earlier neural recipes and development runs used this approach. Other choices also differed, so those results do not isolate its benefit. |
| **Fixed normalization of critic inputs** | Estimate observation means and standard deviations from training warmup data, then freeze the transform used by both critics. This may help when state coordinates have very different scales; the neural features remain trainable. | Tried in earlier Reacher experiments together with the next option and paired simulator rollouts. Their individual effects were not isolated. |
| **Fit value at every observed start in a window** | Give each state in a sampled window a target built from its remaining rewards and advantage corrections. This trains the value network at more of the states used by the advantage update, including near termination. | Tried in the same Reacher work. These overlapping targets reuse the existing trajectory; they are not additional independent samples. |

For extra value passes, recompute the live value prediction and terminal penalty
on each pass while retaining the **same stopped target and pre-update cell
responses**. Simply calling the current `value_loss` repeatedly would reuse an
already consumed autograd graph; this option needs an explicit learner change.

An advanced option, if the simulator supports restoring its state and random
generator, is to try two independently sampled actions from the same state
using **the same simulator randomness**. Pair their action-dependent critic
difference with their own response difference. This can reduce noise in the
action comparison, but needs an adapter and replay extension, and both branch
transitions must count toward the acquisition budget.

That branching approach, fixed input normalization and value fitting at all
window starts were used together in an earlier successful noisy Reacher
recipe. It also used other training settings, including two value passes.
The result supports that combination in its tested setting; it does not show
that any one component improves this one-pass reference or a new application.
[The optional-trial record](verification/optional_implementation_trials.json)
identifies the measurements and their scope.

Evaluate an optional change against the unchanged one-pass baseline using the
same training seeds, evaluation bank and charged acquisition budget. Start with
one change at a time and record it explicitly. These choices do not need to be
resolved before running the supplied examples or the first application pilot.

For implementation details, start with [advg_reference/learner.py](advg_reference/learner.py)
and [examples/train.py](examples/train.py). [VERIFICATION.md](VERIFICATION.md)
records tests and individual outcomes; [PROVENANCE.md](PROVENANCE.md) records the
relationship to the paper and earlier releases. [HANDOFF.md](HANDOFF.md) is a
short collaborator note. [RELEASE.md](RELEASE.md) explains how to share this
standalone package on GitHub. The authors have not yet supplied a license or
citation.
