# Adapting the common recipe

Run an included example first. Then keep `configs/algorithm.json` unchanged for
the first new-task pilot and write the environment adapter. The task file must
contain only `task`, `seed`, `dt`, `microsteps`, `horizon` and `noise_std`; it
cannot select another critic loss, target rule or actor update order.

## What the adapter supplies

1. An observation containing the state needed for a Markov transition and the
   task's time information.
2. A deterministic action with explicit bounds, and a transition after that
   action has actually been executed.
3. Reward integrated over the observation interval, the actual next observation
   and a true terminal flag. Convert a reward rate to an integrated reward once.

`agent.act(observations, explore=True)` returns executed clipped actions.
Add the resulting transitions to `agent.replay` and follow the warmup/update
interval in `examples/train.py`. Flat float32 observations/actions are the
current replay interface. The included finite-horizon task adds three clock
coordinates at the end of each observation and assumes zero terminal value.
For another time encoding, change the observation adapter and the boundary
construction in `AdvG.value_loss` together. Terminal payoff requires matching
reward timing: it can be paid in the final transition's reward while retaining
zero remaining terminal value. If it is represented as a nonzero terminal
value instead, change the terminal targets and boundary handling together.

The optional `actor_network`, `value_network` and `advantage_network` arguments
to AdvG accept learned PyTorch modules. They map observation to action logits,
observation to one scalar, and concatenated observation/action to one scalar,
respectively. The actor wrapper applies the action bounds. Use disjoint,
trainable parameters and deterministic forwards without dropout or batch
normalization. Custom modules retain their initialization; the default F
output head is initialized to zero. Structured/image observations require an
encoder and replay adapter suited to their size.

## The diffusion pilot

Specify what the actor controls in the diffusion project—for example, a drift
modification or guidance coefficient—and how path quality becomes a reward.
Define the state including conditioning and diffusion time, the direction and
units of the schedule, and the terminal condition. These choices determine the
environment interface; they do not require choosing between AdvG variants.

Start with a small neural model and trajectories that can be inspected. Check
time coordinates, reward units, action clipping, terminal bootstrap and action
derivatives on a fixed batch. Run the supplied recipe alongside a simple
baseline with an equal acquisition budget. Record every training seed and use
a fixed evaluation bank; do not choose the best intermediate checkpoint.

The supplied settings provide a concrete first run. If that run fails, diagnose
reward scaling, state/action units and transition handling before tuning. Any
later tuning should be recorded as a new shared configuration, with development
and final evaluation separated. A common recipe reduces implementation choices;
it does not guarantee identical performance on every new task or seed.

## Trying an optional refinement

The README's optional refinements are documented experiments rather than
configuration switches in this package. Keep the supplied one-pass learner as
the baseline and make each learner or adapter extension separately.

For repeated value fits, construct the stopped target and local responses once
before either critic changes. For each value pass, perform a fresh forward
evaluation of `V(states[:, 0])` and of the terminal-boundary observations,
then take an optimizer step against the original target. Update the advantage
critic once using the original stopped local responses, followed by the usual
actor step. The current `value_loss` uses the first live prediction stored in
`terms`, so it cannot be called repeatedly after backward without this change.

For a slowly updated value target, initialize the copy from the learned value
network, exclude its parameters from optimizers and gradients, and specify
which value copy supplies bootstrap targets and local responses. Use the same
chosen copy at both endpoints of each local response. Record the averaging
coefficient and synchronization order; introducing this copy changes the
current-value recipe.

For fixed input normalization, estimate statistics from training warmup only
and freeze them before fitting. Apply the same transform wherever a critic is
evaluated, including terminal-boundary observations and the action derivative
used by the actor. Keep clock construction in raw observation coordinates and
normalize inside the critic. The earlier trial left actor inputs and action
units unchanged. A transform wrapper must preserve the existing network input
and output shapes.

For value fitting at every window start, form stopped targets backward from
the final bootstrap using the recorded reward and advantage correction for
each cell. Average the value losses over observed starts and retain terminal
masks. This reuses a correlated trajectory and increases fitting work; it does
not acquire more independent transitions.

Paired simulator branches require a larger extension: acquire two independent
actions from the same state with shared simulator randomness, store both
successors and rewards, and change the paired advantage response accordingly.
The earlier experiment continued the main branch for its multi-step value
trajectory. Charge both branches and label the result as a simulator-assisted
variant. Its earlier combined results are summarized in
[the optional-trial record](verification/optional_implementation_trials.json).
