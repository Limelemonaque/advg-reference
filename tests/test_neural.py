from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from advg_reference import AdvG, load_recipe
from advg_reference.config import TASK_FIELDS
from advg_reference.tasks import ParallelTasks
from examples.train import run

ROOT = Path(__file__).resolve().parents[1]


def recipe(task='lq', dt=.02):
    return replace(load_recipe(ROOT/f'configs/{task}.json', ROOT/'configs/algorithm.json'), dt=dt)


def learner(cfg):
    dim, action_dim, bound = (7,2,5.) if cfg.task == 'CoupledLQ-v1' else (6,1,2.)
    torch.manual_seed(cfg.seed)
    return AdvG(dim, np.full(action_dim,-bound,np.float32), np.full(action_dim,bound,np.float32), cfg)


def batch_for(agent, cells=3):
    torch.manual_seed(51)
    states = torch.randn(4,cells+1,agent.observation_dim)
    states[..., -3:] = torch.tensor([.2,.3,.4])
    means = agent.actor(states[:,:-1]).detach()
    return dict(states=states, actions=(means+.1*torch.randn_like(means)).detach(), means=means,
                rewards=-.1*torch.rand(4,cells), dones=torch.zeros(4,cells))


def test_tasks_share_every_algorithm_setting():
    left, right = asdict(recipe('lq')), asdict(recipe('pendulum'))
    assert {k:v for k,v in left.items() if k not in TASK_FIELDS} == {k:v for k,v in right.items() if k not in TASK_FIELDS}
    for removed in ('actor_order', 'instrument', 'random_windows', 'value_passes', 'target_tau'):
        assert removed not in left


def test_task_file_cannot_override_algorithm(tmp_path):
    task = json.loads((ROOT/'configs/lq.json').read_text())
    task['target_tau'] = 1.
    path = tmp_path/'task.json'
    path.write_text(json.dumps(task))
    with pytest.raises(ValueError, match='task file'):
        load_recipe(path, ROOT/'configs/algorithm.json')


def test_mesh_changes_preserve_physical_recipe():
    coarse, fine = recipe(), recipe(dt=.005)
    assert fine.window_cells == 4*coarse.window_cells
    assert fine.train_freq == 4*coarse.train_freq
    assert fine.total_timesteps == 4*coarse.total_timesteps
    assert (fine.total_timesteps-fine.warmup_steps)//(fine.num_envs*fine.train_freq) == 12250
    assert fine.actor_hold_duration == coarse.actor_hold_duration and fine.actor_lr == coarse.actor_lr


@pytest.mark.parametrize('index,task,dt', [(0,'lq',.02),(1,'lq',.005),(2,'pendulum',.02),(3,'pendulum',.005)])
def test_single_value_path_matches_preserved_one_step_reference(index,task,dt):
    cfg = replace(recipe(task,dt),actor_hold_duration=0.)
    agent = learner(cfg)
    rng = np.random.default_rng(443)
    obs = rng.normal(size=(cfg.num_envs,agent.observation_dim)).astype(np.float32)
    for _ in range(max(64,cfg.window_cells+4)):
        taken = agent.act(obs,explore=True)
        following = rng.normal(size=obs.shape).astype(np.float32)
        rewards = (-.01*np.square(obs[:,0])).astype(np.float32)
        agent.replay.add(obs,taken,rewards,following,np.zeros(cfg.num_envs,bool))
        obs = following
    for _ in range(3):
        agent.update()
    with np.load(ROOT/'verification/single_value_update_fixtures.npz') as expected:
        for network, net in agent.networks.items():
            for key,value in net.state_dict().items():
                torch.testing.assert_close(value.cpu(),torch.from_numpy(expected[f'r{index}__{network}__{key}']),rtol=2e-5,atol=2e-7)
    assert agent.actor_updates == 3 and agent.optimizer_steps == 9


@pytest.mark.parametrize('task',('lq','pendulum'))
def test_online_networks_are_fully_trainable(task):
    agent = learner(recipe(task))
    for name in ('actor','value','q'):
        params = list(agent.networks[name].parameters())
        assert all(p.requires_grad for p in params)
        assert {id(p) for p in params} == {id(p) for g in agent.optimizers[name].param_groups for p in g['params']}
        assert sum(isinstance(m,nn.Linear) for m in agent.networks[name].modules()) == 3
    assert set(agent.networks) == {'actor','value','q'}


@pytest.mark.parametrize('task',('lq','pendulum'))
def test_pairing_has_no_cross_cell_response(task):
    agent = learner(recipe(task))
    with torch.no_grad():
        agent.q.network[-1].weight.fill_(.2)
    batch = batch_for(agent)
    terms = agent.critic_terms(batch)
    gradients = torch.autograd.grad(agent.advantage_loss(terms),tuple(agent.q.parameters()))
    changed = dict(batch,rewards=batch['rewards'].clone())
    changed['rewards'][0,1] += 2.
    other = agent.critic_terms(changed)
    other_gradients = torch.autograd.grad(agent.advantage_loss(other),tuple(agent.q.parameters()))
    own = agent.critic_terms(batch)
    factor = 2*agent.config.dt
    predicted = torch.autograd.grad(-2*factor*own['weights'][1]*own['instrument'][0,1]/4,tuple(agent.q.parameters()))
    for before,after,expected in zip(gradients,other_gradients,predicted):
        torch.testing.assert_close(after-before,expected,rtol=1e-4,atol=2e-8)


def test_anchored_critic_and_required_stops():
    agent = learner(recipe())
    terms = agent.critic_terms(batch_for(agent))
    assert not terms['target'].requires_grad and not terms['local'].requires_grad
    gradient, = torch.autograd.grad(terms['q'].sum(),agent.q.network[-1].bias,retain_graph=True)
    assert torch.equal(gradient,torch.zeros_like(gradient))
    unused = torch.autograd.grad(agent.advantage_loss(terms),tuple(agent.actor.parameters())+tuple(agent.value.parameters()),allow_unused=True)
    assert all(item is None for item in unused)


@pytest.mark.parametrize('task',('lq','pendulum'))
def test_one_value_step_frozen_response_and_post_critic_actor(task):
    agent = learner(recipe(task))
    batch = batch_for(agent)
    old = [p.detach().clone() for p in agent.actor.parameters()]
    targets, order = [], []
    before = agent.critic_terms(batch)
    original_loss = agent.value_loss
    def recorded_loss(terms):
        targets.append(terms['target'].clone())
        return original_loss(terms)
    agent.value_loss = recorded_loss
    original_gradients = agent._gradients
    def recorded_gradients(name,loss):
        order.append(name)
        return original_gradients(name,loss)
    agent._gradients = recorded_gradients
    original_advantage_loss = agent.advantage_loss
    def checked_advantage_loss(terms):
        torch.testing.assert_close(terms['local'],before['local'],rtol=0,atol=0)
        return original_advantage_loss(terms)
    agent.advantage_loss = checked_advantage_loss
    acquired = round(agent.config.actor_hold_duration/agent.config.dt)
    metrics = agent.update(batch,actor_states=batch['states'][:,0],acquired=acquired)
    assert len(targets) == 1 and torch.equal(targets[0],before['target'])
    assert order == ['value','q','actor']
    # A zero initialized F gives no pre-update actor direction; the updated F does.
    assert any(not torch.equal(a,b) for a,b in zip(old,agent.actor.parameters()))
    assert metrics['actor_action_step_rms'] <= agent.actor_cap_at(acquired)*(1+1e-6)
    assert agent.optimizer_steps == 3
    assert all(state['step'] == 1 for optimizer in agent.optimizers.values() for state in optimizer.state.values())


def test_actor_hold_and_physical_cap_schedule():
    agent = learner(recipe())
    old = [p.detach().clone() for p in agent.actor.parameters()]
    agent.update(batch_for(agent),acquired=0)
    assert agent.actor_updates == 0 and agent.optimizer_steps == 2
    assert all(torch.equal(a,b) for a,b in zip(old,agent.actor.parameters()))
    fine = learner(recipe(dt=.005))
    for duration in (0.,400.,800.,1200.,2000.):
        assert agent.actor_cap_at(round(duration/.02)) == fine.actor_cap_at(round(duration/.005))


def test_terminal_bootstrap_and_reset_boundary():
    agent = learner(recipe())
    batch = batch_for(agent)
    batch['dones'][:,-1] = 1.
    original = agent.critic_terms(batch)
    changed = dict(batch,states=batch['states'].clone())
    changed['states'][:,-1] += 1000.
    following = agent.critic_terms(changed)
    torch.testing.assert_close(original['target'],following['target'])
    torch.testing.assert_close(original['local'],following['local'])
    batch['dones'][:,0] = 1.
    with pytest.raises(ValueError,match='crosses'):
        agent.critic_terms(batch)


def test_bad_data_does_not_commit_updates():
    agent = learner(recipe())
    batch = batch_for(agent)
    batch['rewards'][0,0] = torch.nan
    with pytest.raises(ValueError):
        agent.update(batch)
    assert agent.updates == 0


@pytest.mark.parametrize('task',('lq','pendulum'))
def test_collector_stores_true_terminal_observation(task):
    cfg = replace(recipe(task),horizon=.2)
    tasks = ParallelTasks(cfg)
    tasks.reset()
    for _ in range(cfg.episode_steps):
        obs,_,dones,replay_next = tasks.step(np.zeros((cfg.num_envs,tasks.action_space.shape[0]),np.float32))
    assert dones.all() and (replay_next[:,-3] == 1.).all() and (obs[:,-3] == 0.).all()
    tasks.close()


@pytest.mark.parametrize('task,dt',[('lq',.02),('pendulum',.02),('lq',.005),('pendulum',.005)])
def test_standalone_shared_recipe_smoke(task,dt,tmp_path):
    original = recipe(task,dt)
    warmup = max(128,original.window_cells*original.num_envs)
    cfg = replace(original,training_duration=(warmup+128)*dt,warmup_duration=warmup*dt,
        update_interval=original.num_envs*dt,buffer_duration=1024*dt,
        evaluation_interval=128*dt,curve_episodes=1,final_episodes=1)
    result = run(cfg,tmp_path/'run')
    assert result['critic_calls'] == 32 and result['actor_updates'] == 0 and result['optimizer_steps'] == 64
    assert result['training_transitions'] == cfg.total_timesteps
    with pytest.raises(FileExistsError):
        run(cfg,tmp_path/'run')
