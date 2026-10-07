# The shared neural AdvG update

This update is used unchanged on every task. Only the environment, dimensions,
action bounds, observation time step and finite horizon enter through the task
interface. All numerical training settings come from `configs/algorithm.json`.

## Networks and recorded transitions

The actor `mu(s)` is a tanh-bounded MLP. The value `V(s)` and state-action
network `F(s,a)` are scalar MLPs. Every network has two width-128 ReLU hidden
layers. All online parameters are trainable and disjoint. F's output head
starts at zero; the hidden layers are trained along with that head.

The advantage rate compares an action with the current actor's action:

```text
q(s,a) = F(s,a) - F(s,stop(mu(s))).
```

Both F evaluations retain critic-parameter derivatives. The actor anchor is
stopped during critic fitting. There is no separate target-value network.

Collection adds clipped Gaussian noise of standard deviation 0.3 to the actor's
action. Replay stores detached, consecutive observations `states[B,L+1,S]`,
executed actions `actions[B,L,A]`, integrated rewards `rewards[B,L]` and true
terminal flags `dones[B,L]`. It stores the actual terminal successor before an
environment resets. An arbitrary collection cutoff should retain bootstrap;
a true terminal suppresses it.

One window length L is drawn uniformly from 1 through `round(0.2/dt)` for each
minibatch. Eligible starts are sampled from FIFO replay with replacement.
Windows may overlap or be reused but cannot cross an episode reset. A true
terminal can occur only in the final cell of a sampled window.

## Freeze the responses, then fit the critics

Write `h=dt` and `gamma=exp(-discount_rate*h)`. Before any parameter changes,
form the stopped multi-step value target and stopped local cell responses:

```text
T = gamma**L*(1-done[L-1])*V_old(s[L])
    + sum_i gamma**i*(R[i] - h*stop(q_old[i]))

D[i] = h*stop(q_old[i])
       - (R[i] + gamma*(1-done[i])*V_old(s[i+1]) - V_old(s[i])).
```

The term in parentheses compares reward plus continuation value with the value
at the cell's start. The predicted advantage contribution is `h*q_old[i]`.
Their difference is the response D paired with that cell's action dependence.

Update V once against the **stopped T**. This update minimizes
`0.5*mean((V(s[0])-T)**2)` plus the zero-terminal-value penalty
`0.5*terminal_weight*mean(V(terminal_clock(s[0]))**2)`.
The examples' last three observation coordinates are
`[elapsed_fraction, cos(2*pi*elapsed_fraction), sin(2*pi*elapsed_fraction)]`;
`terminal_clock` replaces those with `[1,1,0]` while retaining physical state.

Fit F once by minimizing the semi-gradient surrogate

```text
2*h * mean(sum_i gamma**i * q_live[i]*stop(D[i])).
```

Each cell is multiplied by its own stopped response before aggregation; there
are no products between one cell's action instrument and another cell's
response. D and T remain at the pre-update values throughout all critic fits.
This surrogate is not a squared TD error, so its scalar value need not decrease
monotonically or approach zero. Each critic receives one optimizer step per
call to `update()`.

## Improve the actor with the updated critic

After the critic fits, sample actor states separately from replay. Once the
shared actor start time is reached, minimize

```text
-h * mean(F_updated(s,mu(s))).
```

F's parameters are frozen for this backward pass while gradients propagate
through its action input to the actor. This is the deterministic policy
gradient using the newly updated critic. The action-independent anchor in q
does not change its action derivative.

V and F use Adam at 3e-4, and the actor uses Adam at 1e-4. Gradient norms are
clipped at 10. An actor proposal is interpolated toward its previous parameters
until the RMS action displacement on sampled states, divided by the action
half-range, satisfies the shared cap. Adam's internal state is retained when
its parameter proposal is reduced.

## Common timing

Training uses four serial environments. Acquisition durations below sum time
over all four environments; episode horizons and critic windows describe an
individual trajectory.

| Setting | Shared value |
|---|---:|
| Total training acquisition | 2000 physical time units |
| Replay capacity | 400 physical time units |
| Collection warmup | 40 physical time units |
| Acquisition between critic calls | 0.16 physical time units |
| Actor start | 400 physical time units |
| Actor cap | 0.0005 → 0.003 between acquisition times 400 and 1200 |
| Minibatch | 64 windows |

At h=0.02 these imply 100,000 acquired transitions, warmup 2,000, 12,250 critic
calls and 10,001 actor updates: 34,501 optimizer steps in total. At h=0.005
they imply 400,000 transitions, warmup 8,000 and the same update counts.
Simulator, exploratory-action, replay
and window-length random streams are separate. The learner refuses to continue
after an incomplete failed update.
