# Verification of the one-value-update recipe

Version 0.4.0 takes one value update per critic call. The recorded environment
is Windows with Python 3.12.10, CPU Torch 2.5.1+cpu, NumPy 1.26.4, SciPy 1.14.1,
Gymnasium 1.0.0 and pytest 8.3.5. Direct dependency versions are pinned in
`requirements-tested.txt`.

## Implementation checks

The 24 tests cover shared configuration and rejected task overrides, trainable
neural parameters, exact saved optimizer fixtures at `dt=0.02` and `dt=0.005`,
cell pairing, required gradient stops, one optimizer step per critic, responses
frozen before either critic changes, post-critic actor updates, the actor
start/cap schedule, terminal bootstrap and actual terminal successor storage.
Short standalone runs cover both tasks and both meshes.

`verification/single_value_update_parity.json` compares the learner against the
archived v0.3.0 implementation with only its second, third and fourth value
optimizer calls skipped. Through three training calls, all online parameters
and Adam states agree bitwise on both tasks and both meshes. Actor hold is
disabled only in these fixtures to exercise the actor. The normal configuration
retains the common start time. Current portable fixtures are in
`verification/single_value_update_fixtures.npz`; tests allow a small numerical
tolerance across runtime environments.

The historical v0.3.0 fixtures and parity record retain their original filenames
`shared_update_fixtures.npz` and `update_parity.json`. They describe the four-pass
version and are not the fixtures used by the current tests.

## Matched training comparison

Only the number of value optimizer steps changes between the versions below.
Both use the same two training seeds per task, all numerical settings, 100,000
training transitions per seed at `dt=0.02`, and 50 final evaluation episodes
with seed base 9790000. Initial policies are also evaluated on that same bank.
The reported result uses the final policy, without selecting an intermediate
checkpoint. Higher discounted return is better.

| Task | Training seed | Initial return | Four value updates (v0.3.0) | One value update (v0.4.0) |
|---|---:|---:|---:|---:|
| LQ | 20801 | -8.19284 | -3.86716 | -4.07175 |
| LQ | 20802 | -7.96452 | -3.65196 | -4.04242 |
| Pendulum | 20801 | -18.04555 | -9.66562 | -7.85649 |
| Pendulum | 20802 | -18.02478 | -7.96573 | -8.14255 |

The one-update means are -4.05709 on LQ and -7.99952 on Pendulum, compared with
-3.75956 and -8.81567 for four updates. All four new runs improve over their
initial policies. The one-update recipe is slightly worse on LQ and better on
Pendulum on average in these checks. With two training seeds per task, these
measurements support the simpler reference but do not establish a general
ranking or diffusion-model transfer.

Each new run makes 12,250 value steps, 12,250 advantage steps and 10,001 actor
steps: **34,501 optimizer steps**, compared with 71,251 in the four-pass version.
The training acquisition and actor-update counts are unchanged. This is a
reduction in optimizer steps, not a controlled wall-clock speed comparison.

`verification/single_value_benchmarks.json` contains the new outcomes,
source/configuration hashes, per-episode final returns and matched differences.
It also identifies the unchanged baseline record
`verification/shared_benchmarks.json`, which retains the v0.3.0 results and its
earlier development screens. The four new runs acquired 400,000 training and
224,000 evaluation transitions, including initial-policy evaluations on the
final bank. Earlier training and evaluation are not counted as new acquisition.

Full training was checked at `dt=0.02`. The finer mesh is covered by optimizer
fixtures and short execution checks; full fine-mesh training is not claimed.

## Standalone packaging

`verification/portable_checks.json` records testing of the v0.4.0 archive
extracted outside the parent source tree, including the README's complete loop,
the test suite, CLI smoke runs and a wheel build. This uses the existing
dependency environment; it is not a fresh installation or cross-platform check.
The historical portable record is retained as `v0_3_portable_checks.json`.
Packaging tests check per-file manifests, excluded outputs, deterministic bytes
and refusal to overwrite an archive.
