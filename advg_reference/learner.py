"""Neural DDPG-style AdvG with multi-step value TD and cell-paired advantage fitting.

Both critics and the actor are fully trainable neural networks. One critic-then-actor update is used for every task. There are no task-specific
algorithm branches, lagged targets or ghost actions.
"""
from __future__ import annotations
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .replay import VectorReplay
from .config import Recipe


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width),
                         nn.ReLU(), nn.Linear(width, outputs))


class BoundedActor(nn.Module):
    def __init__(self, observation_dim, low, high, width, init_scale, network=None):
        super().__init__()
        low, high = torch.as_tensor(low), torch.as_tensor(high)
        self.register_buffer("center", (high + low) / 2)
        self.register_buffer("scale", (high - low) / 2)
        self.network = mlp(observation_dim, low.numel(), width) if network is None else network
        if network is None and init_scale > 0:
            nn.init.uniform_(self.network[-1].weight, -init_scale, init_scale)
            nn.init.zeros_(self.network[-1].bias)

    def logits(self, observations):
        return self.network(observations)

    def forward(self, observations):
        return self.center + self.scale * torch.tanh(self.logits(observations))


class ActionCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, width, network=None):
        super().__init__()
        self.network = mlp(observation_dim + action_dim, 1, width) if network is None else network

    def forward(self, observations, actions):
        return self.network(torch.cat((observations, actions), dim=-1)).squeeze(-1)


class NeuralReplay(VectorReplay):
    """One uniform window length per minibatch; the same rule for every task."""
    def __init__(self, cfg, observation_space, action_space):
        super().__init__(cfg, observation_space, action_space)
        self.length_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 958417]))

    def sample_windows(self, batch_size, cells, **kwargs):
        length = int(self.length_rng.integers(1,cells+1))
        return super().sample_windows(batch_size,length,**kwargs)


class AdvG:
    """Fully trainable neural critics; multi-step TD, paired q update, deterministic actor.

    Optional network arguments replace the learned MLPs, not the update rule.
    They map observations -> action logits, observations -> scalar [...,1],
    and concatenated (observations, actions) -> scalar [...,1], respectively.
    Modules must have disjoint trainable parameters and deterministic forwards.
    The example runner handles finite-horizon clocks and true terminal storage.
    """
    def __init__(self, observation_dim, low, high, recipe: Recipe, *, device="cpu",
                 actor_network=None, value_network=None, advantage_network=None):
        self.config = recipe
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.low, self.high = np.asarray(low, np.float32), np.asarray(high, np.float32)
        if self.low.ndim != 1 or self.low.shape != self.high.shape or not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or np.any(self.low >= self.high):
            raise ValueError("finite nondegenerate vector action bounds required")
        self.action_dim = len(self.low)
        # Keep constructor order and RNG consumption equal to the paper code.
        self.actor = BoundedActor(observation_dim, self.low, self.high, recipe.width,
                                  recipe.actor_init_scale, actor_network).to(self.device)
        self.value = (mlp(observation_dim, 1, recipe.width) if value_network is None else value_network).to(self.device)
        self.q = ActionCritic(observation_dim, self.action_dim, recipe.width, advantage_network).to(self.device)
        if advantage_network is None:
            nn.init.zeros_(self.q.network[-1].weight)
            nn.init.zeros_(self.q.network[-1].bias)
        self.networks = dict(actor=self.actor, value=self.value, q=self.q)
        params = [p for name in ("actor", "value", "q") for p in self.networks[name].parameters()]
        if len({id(p) for p in params}) != len(params) or any(not p.requires_grad for p in params):
            raise ValueError("actor and critics need disjoint, fully trainable parameters")
        if any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd))
               for name in ("actor", "value", "q") for m in self.networks[name].modules()):
            raise ValueError("use deterministic networks without dropout or batch normalization")
        self.optimizers = {name: torch.optim.Adam(self.networks[name].parameters(), lr=rate)
                           for name, rate in (("actor", recipe.actor_lr), ("value", recipe.critic_lr), ("q", recipe.critic_lr))}
        spaces = [SimpleNamespace(shape=(observation_dim,)), SimpleNamespace(shape=(self.action_dim,))]
        self.replay = NeuralReplay(recipe, *spaces)
        self.action_rng = np.random.default_rng(np.random.SeedSequence([recipe.seed, 91019]))
        self.updates = self.actor_updates = self.optimizer_steps = 0
        self._healthy = True

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def act(self, observations, *, explore=False):
        with torch.no_grad():
            means = self.actor(self.tensor(observations)).cpu().numpy()
        action = means.copy()
        if explore:
            action = action.astype(np.float64) + self.config.exploration_std * self.action_rng.normal(size=means.shape)
        return np.clip(action, self.low, self.high).astype(np.float32)

    def critic_terms(self, batch):
        cfg = self.config
        states, actions = self.tensor(batch["states"]), self.tensor(batch["actions"])
        rewards, dones = self.tensor(batch["rewards"]), self.tensor(batch["dones"])
        if states.ndim != 3 or actions.ndim != 3 or rewards.ndim != 2:
            raise ValueError("expected states[B,L+1,S], actions[B,L,A], rewards[B,L]")
        b, length = actions.shape[:2]
        if (b < 1 or length < 1 or states.shape != (b, length + 1, self.observation_dim)
                or actions.shape[-1] != self.action_dim or rewards.shape != (b, length) or dones.shape != (b, length)):
            raise ValueError("inconsistent window shapes")
        if not all(torch.isfinite(t).all() for t in (states, actions, rewards, dones)) or not torch.isin(dones, self.tensor([0, 1])).all():
            raise ValueError("nonfinite data or invalid terminal flags")
        if dones[:, :-1].any():
            raise ValueError("window crosses a terminal/reset")
        if any(t.requires_grad for t in (states, actions, rewards, dones)):
            raise ValueError("recorded replay data must be detached")
        weights = cfg.gamma ** torch.arange(length, dtype=torch.float32, device=self.device)
        flat = states[:, :-1].reshape(-1, self.observation_dim)
        taken = actions.reshape(-1, self.action_dim)
        with torch.no_grad():
            anchor = self.actor(flat)
            target_values = self.value(states.reshape(-1,self.observation_dim)).reshape(b,length+1)
            local_values = target_values
        q = (self.q(flat, taken) - self.q(flat, anchor)).reshape(b, length)
        value_start = self.value(states[:, 0]).squeeze(-1)
        with torch.no_grad():
            right = target_values[:, 1:] * (1 - dones)
            target = cfg.gamma ** length * right[:, -1] + (weights * (rewards - cfg.dt * q.detach())).sum(1)
            local = local_values[:, :-1] - cfg.gamma * (1 - dones) * local_values[:, 1:] - rewards + cfg.dt * q.detach()
        instrument = q
        return dict(states=states, q=q, instrument=instrument, local=local, weights=weights,
                    value_start=value_start, target=target, residual=(value_start.detach() - target).detach())

    def advantage_loss(self, terms):
        cfg = self.config
        paired = (terms["weights"] * terms["instrument"] * terms["local"].detach()).sum(1).mean()
        return 2*cfg.dt*paired

    def value_loss(self, terms):
        start = terms["value_start"]
        boundary = terms["states"][:, 0].detach().clone()
        boundary[:, -3:] = self.tensor([1., 1., 0.])
        return (.5 * (start - terms["target"]).square().mean()
                + .5 * self.config.terminal_weight * self.value(boundary).square().mean())

    def _gradients(self, name, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {name} objective")
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for p in self.networks[name].parameters() if p.grad is not None]
        norm = sum(p.grad.detach().double().square().sum() for p in parameters).sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError(f"nonfinite {name} gradient")
        nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm, error_if_nonfinite=True)
        return float(norm)

    def _step(self, name, loss):
        norm = self._gradients(name, loss)
        self.optimizers[name].step()
        self.optimizer_steps += 1
        return norm

    def actor_cap_at(self, acquired):
        cfg = self.config
        physical_time = acquired*cfg.dt
        fraction = min(1.,max(0.,(physical_time-cfg.cap_ramp_start)/(cfg.cap_ramp_end-cfg.cap_ramp_start)))
        return cfg.actor_cap+fraction*(cfg.actor_cap_final-cfg.actor_cap)

    def _commit_actor(self, states, acquired):
        old = [p.detach().clone() for p in self.actor.parameters()]
        with torch.no_grad():
            old_actions = self.actor(states).clone()
        self.optimizers['actor'].step()
        self.optimizer_steps += 1
        self.actor_updates += 1
        cap = self.actor_cap_at(acquired)
        with torch.no_grad():
            proposal = [p.detach().clone() for p in self.actor.parameters()]
            def displacement():
                return float(((self.actor(states) - old_actions) / self.actor.scale).square().mean().sqrt())
            realized = displacement()
            fraction = 1.
            if cap and realized > cap:
                fraction = cap / realized
                for _ in range(13):
                    for p, initial, target in zip(self.actor.parameters(), old, proposal):
                        p.copy_(initial + fraction * (target - initial))
                    realized = displacement()
                    if realized <= cap * (1 + 1e-6):
                        break
                    fraction *= .5
                if realized > cap * (1 + 1e-6):
                    for p, initial in zip(self.actor.parameters(), old):
                        p.copy_(initial)
                    realized, fraction = 0., 0.
        return realized, fraction

    def update(self, batch=None, *, actor_states=None, acquired=None):
        if not self._healthy:
            raise RuntimeError('an update failed; discard/reload the learner before continuing')
        cfg = self.config
        acquired = self.replay.total_transitions if acquired is None else acquired
        if batch is None:
            batch = self.replay.sample_windows(cfg.batch_size,cfg.window_cells)
        # Freeze all responses and the multi-step target before fitting either critic.
        terms = self.critic_terms(batch)
        self._healthy = False
        norms = dict(value=self._step('value',self.value_loss(terms)))
        norms['q'] = self._step('q',self.advantage_loss(terms))
        actor_norm = movement = 0.
        if acquired*cfg.dt >= cfg.actor_hold_duration:
            actor_batch = self.tensor(self.replay.sample_states(cfg.batch_size) if actor_states is None else actor_states)
            self.q.requires_grad_(False)
            try:
                actor_loss = -cfg.dt*self.q(actor_batch,self.actor(actor_batch)).mean()
                actor_norm = self._gradients('actor',actor_loss)
                movement,_ = self._commit_actor(actor_batch,acquired)
            finally:
                self.q.requires_grad_(True)
        self.updates += 1
        if any(not torch.isfinite(p).all() for net in self.networks.values() for p in net.parameters()):
            raise FloatingPointError('nonfinite parameter after update')
        self._healthy = True
        return dict(update=self.updates,actor_updates=self.actor_updates,optimizer_steps=self.optimizer_steps,
                    value_gradient_norm=norms['value'],advantage_gradient_norm=norms['q'],
                    actor_gradient_norm=actor_norm,actor_action_step_rms=movement,
                    cell_residual_rms=float(terms['local'].double().square().mean().sqrt()),
                    window_residual_rms=float(terms['residual'].double().square().mean().sqrt()),
                    sampled_cells=int(np.prod(batch['actions'].shape[:2])))

    def parameters_state(self):
        return {name: deepcopy(net.state_dict()) for name, net in self.networks.items()}
