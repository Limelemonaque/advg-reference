# Provenance

Version 0.4.0, prepared on 2026-10-07, simplifies the shared neural recipe at
the author's request. Each training call now takes one value-critic optimizer
step, one advantage-critic step and, after the actor start time, one actor step.
Version 0.3.0 took four value steps per call. The extra three value steps have
been removed; the networks, losses, pre-update responses, post-critic actor,
replay rule, acquisition budget and numerical settings are retained.

## Source lineage

The original neural update comes from `PostOrderPracticalAgent` in
`output/iclr/canonical_baselines_20260925/source/iclr_experiments/advg_practical_post_order.py`
and the corresponding fine-mesh suite in the parent research workspace. These
are provenance identifiers, not runtime dependencies. Version 0.3.0 extracted
that post-critic/current-value update and applied one recipe to LQ and Pendulum,
removing a redundant value copy synchronized after every update.

Version 0.4.0 is a subsequent simplification of that reference. It is not an
exact reproduction of the paper's four-pass Pendulum training schedule.
`verification/single_value_update_parity.json` records checks against the
archived v0.3.0 implementation with only its three extra value optimizer calls
skipped. Online parameters and Adam state agree bitwise through three training
calls on both tasks and both time meshes. Portable optimizer fixtures come
from that independently adapted archived implementation.

The LQ and Pendulum dynamics, eligible replay sampler, finite-horizon clock
encoding and true terminal successor handling retain their frozen-source
lineage. LQ matrices enter only the simulator, not critic features or training
labels. All learner features are trainable neural layers.

## Experiments and earlier releases

The v0.4.0 single-pass measurements are recorded in
`verification/single_value_benchmarks.json`. The v0.3.0 four-pass measurements
remain unchanged in `verification/shared_benchmarks.json`. The comparison uses
the same two training seeds per task, acquisition budgets and final evaluation
bank; each version retains its actual source hashes. Historical four-pass
outcomes are not assigned to the single-pass learner.

`verification/shared_update_fixtures.npz` and `verification/update_parity.json`
are historical v0.3.0 checks. Current tests use the separately named
`single_value_update_fixtures.npz`. The v0.3.0 archive, including its later README
refresh, remains unchanged.

The v0.2 archive preserved each paper task's distinct recipe. The v0.1
fixed-feature demonstration is also superseded. Their sources, measurements
and archives remain separate from the common neural reference.

The README's optional refinements are documented extensions. Their existing
trial evidence is summarized in `verification/optional_implementation_trials.json`,
including a separate simulator-assisted Reacher recipe. Those historical bundle
results do not establish individual component gains or gains over v0.4.0.
Documenting the choices adds no switches to the current one-pass learner.

The manuscript's finite-feature theoretical specialization does not establish
convergence of this neural Adam/replay training loop. No manuscript or historical
result is changed by this release. No CT-DDPG baseline is bundled; if one is
added, unqualified CT-DDPG denotes the released full-window gradient, while the
stopped start-of-window semi-gradient is a separately named comparator.

Author attribution, a citation and a software license remain author decisions.
Dependency source is not vendored. The archive excludes parent manuscripts,
large checkpoints and exploratory logs.
