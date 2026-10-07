"""Run a recorded neural AdvG recipe; no parent-repository imports or fixed features."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import scipy
import gymnasium

from advg_reference import AdvG, Recipe, load_recipe, __version__
from advg_reference.tasks import FiniteHorizonTask, ParallelTasks


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def evaluate(agent, cfg, episodes, seed_base):
    env = FiniteHorizonTask(cfg, seed_base)
    rows = []
    try:
        for seed in range(seed_base, seed_base + episodes):
            obs = env.reset(seed=seed)
            discounted = integrated = 0.
            for step in range(cfg.episode_steps):
                action = agent.act(obs)
                obs, reward, done = env.step(action)
                discounted += cfg.gamma ** step * reward
                integrated += reward
                if done:
                    break
            rows.append(dict(seed=seed, discounted_return=discounted, integrated_return=integrated, transitions=step+1))
    finally:
        env.close()
    return dict(discounted_return=float(np.mean([r['discounted_return'] for r in rows])),
                integrated_return=float(np.mean([r['integrated_return'] for r in rows])),
                episodes=rows, transitions=sum(r['transitions'] for r in rows))


def run(cfg, output):
    cfg.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.seed)
    root = Path(__file__).resolve().parents[1]
    hashes = {}
    for folder in ('advg_reference', 'examples'):
        for path in sorted((root / folder).glob('*.py')):
            relative = path.relative_to(root)
            data = path.read_bytes()
            dest = output / 'source_snapshot' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    write_json(output / 'manifest.json', dict(version=__version__, recipe=asdict(cfg), source_sha256=hashes,
        runtime=dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                     numpy=np.__version__, scipy=scipy.__version__, gymnasium=gymnasium.__version__),
        scope='one shared neural recipe on every task; new experiments, separate from paper runs'))
    tasks = ParallelTasks(cfg)
    agent = AdvG(tasks.observation_space.shape[0], tasks.action_space.low, tasks.action_space.high, cfg)
    obs = tasks.reset()
    start = time.perf_counter()
    eval_transitions = 0
    history = []
    try:
        with (output / 'evaluations.jsonl').open('w', encoding='utf-8') as eval_log, (
                output / 'diagnostics.jsonl').open('w', encoding='utf-8') as diagnostic_log:
            for count in range(0, cfg.total_timesteps + 1, cfg.num_envs):
                if count == 0 or count % cfg.eval_every == 0 or count == cfg.total_timesteps:
                    row = evaluate(agent, cfg, cfg.curve_episodes, cfg.curve_seed_base)
                    row.update(training_transitions=count, physical_time=count*cfg.dt)
                    eval_transitions += row['transitions']
                    history.append(row)
                    eval_log.write(json.dumps(row, allow_nan=False) + '\n')
                    print(json.dumps({k: row[k] for k in ('training_transitions', 'discounted_return')}), flush=True)
                if count == cfg.total_timesteps:
                    break
                actions = agent.act(obs, explore=True)
                following, rewards, dones, replay_following = tasks.step(actions)
                agent.replay.add(obs, actions, rewards, replay_following, dones)
                obs = following
                acquired = count + cfg.num_envs
                if (acquired > cfg.warmup_steps and (acquired-cfg.warmup_steps) % (cfg.num_envs*cfg.train_freq) == 0
                        and agent.replay.filled_vector_steps >= cfg.window_cells):
                    metrics = agent.update()
                    if agent.updates == 1 or agent.updates % 100 == 0 or acquired == cfg.total_timesteps:
                        diagnostic_log.write(json.dumps(dict(training_transitions=acquired, **metrics), allow_nan=False) + '\n')
        final = evaluate(agent, cfg, cfg.final_episodes, cfg.final_seed_base)
        eval_transitions += final['transitions']
        write_json(output / 'final_evaluation.json', final)
        torch.save(agent.parameters_state(), output / 'parameters.pt')
        expected_calls = (cfg.total_timesteps-cfg.warmup_steps) // (cfg.num_envs*cfg.train_freq)
        if agent.updates != expected_calls:
            raise RuntimeError('actual critic-call count differs from the recipe schedule')
        result = dict(status='completed', seed=cfg.seed, task=cfg.task, dt=cfg.dt, training_transitions=cfg.total_timesteps,
            training_physical_time=cfg.total_timesteps*cfg.dt, evaluation_transitions=eval_transitions,
            critic_calls=agent.updates, actor_updates=agent.actor_updates, optimizer_steps=agent.optimizer_steps,
            initial_curve_return=history[0]['discounted_return'], final_curve_return=history[-1]['discounted_return'],
            final_heldout_return=final['discounted_return'], elapsed_seconds=time.perf_counter()-start,
            parameter_counts={name: sum(p.numel() for p in net.parameters()) for name, net in agent.networks.items()})
        write_json(output / 'result.json', result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        tasks.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/lq.json')
    parser.add_argument('--algorithm', default='configs/algorithm.json')
    parser.add_argument('--dt', type=float, help='change the observation mesh; physical settings stay fixed')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true', help='explicitly shorter recipe for execution checks, not paper reproduction')
    args = parser.parse_args()
    cfg = load_recipe(args.config, args.algorithm)
    if args.dt is not None:
        cfg = replace(cfg, dt=args.dt)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.smoke:
        warmup = max(128, cfg.window_cells*cfg.num_envs)
        cfg = replace(cfg, training_duration=(warmup+128)*cfg.dt, warmup_duration=warmup*cfg.dt,
                      update_interval=cfg.num_envs*cfg.dt, buffer_duration=1024*cfg.dt,
                      curve_episodes=2, final_episodes=2, evaluation_interval=128*cfg.dt)
    run(cfg, args.output)


if __name__ == '__main__':
    main()
