import functools

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.task, librl.train.train_loop, librl.train.cc
import pytest
import torch, torch.utils

import nasrl.nn
import nasrl.tree.actor, nasrl.tree.env
import nasrl.replay
import nasrl.field

from . import * 

#######################
# Agent / environment #
#######################
def get_mlp_conf():
    interval = nasrl.field.Interval(10, 1000)
    field = nasrl.field.Field(interval, lambda *_:200)
    return nasrl.field.MLPConf(10, field)
    
def get_cnn_conf():
    from nasrl.field.field import Interval, Field
    from nasrl.field.conf import CNNConf
    # Choose sensible ranges for (c)hannels, (k)ernel size, (s)tride,
    # (p)adding, and (d)ilation. `lambda` repesent sensible initial values.
    c_i, k_i = Interval(1, 128), Interval(2, 8)
    s_i, p_i, d_i = Interval(1, 4), Interval(0, 4), Interval(1, 4)
    c_f, k_f = Field(c_i, lambda *_:4), Field(k_i, lambda *_:4)
    s_f, p_f, d_f = Field(s_i, lambda *_:1), Field(p_i, lambda *_:2), Field(d_i, lambda *_:1)
    return CNNConf(10, c_f, k_f, s_f, p_f, d_f, lambda _: 1)
    
def mlp_helper(mnist_dataset):
    loaders = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    # TODO: If last adapt step stops updating NN, must change this to 1.
    env = nasrl.tree.env.MLPClassificationEnv((1,28,28), get_mlp_conf(), torch.nn.CrossEntropyLoss(), *loaders, adapt_steps=0)
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = nasrl.tree.actor.MLPTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return mnist_dataset.hypers, env, critic_net, policy_net

def cnn_helper(mnist_dataset):
    loaders = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    # TODO: If last adapt step stops updating NN, must change this to 1.
    env = nasrl.tree.env.CNNClassificationEnv((1,28,28), get_cnn_conf(), torch.nn.CrossEntropyLoss(), *loaders, adapt_steps=0)
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = nasrl.tree.actor.CNNTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return mnist_dataset.hypers, env, critic_net, policy_net

def joint_helper(mnist_dataset):
    loaders = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    # TODO: If last adapt step stops updating NN, must change this to 1.
    env = nasrl.tree.env.JointClassificationEnv((1,28,28), get_cnn_conf(), get_mlp_conf(), torch.nn.CrossEntropyLoss(), *loaders, adapt_steps=0)
    # Construct an NN to process MLP and CNN network descriptions.
    cnn_size = functools.reduce(lambda x,y: x*y, env.cnn_observation_space.shape, 1)
    mlp_size = functools.reduce(lambda x,y: x*y, env.mlp_observation_space.shape, 1)
    cnn_policy_kernel = librl.nn.core.MLPKernel(cnn_size)
    mlp_policy_kernel = librl.nn.core.MLPKernel(mlp_size)
    # Use a bi-linear layer to combine state information about the MLP and CNN
    # to properly init cnn/mlp weighs.
    fusion_kernel = librl.nn.core.BilinearKernel(cnn_policy_kernel, mlp_policy_kernel, 10)
    policy_net = nasrl.tree.actor.JointTreeActor(cnn_policy_kernel, mlp_policy_kernel, fusion_kernel, env.observation_space)
    cnn_policy_kernel = librl.nn.core.MLPKernel(cnn_size)
    mlp_policy_kernel = librl.nn.core.MLPKernel(mlp_size)
    critic_kernel = nasrl.nn.BilinearAdapter(cnn_policy_kernel, mlp_policy_kernel, 10)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return mnist_dataset.hypers, env, critic_net, policy_net  

#########
# Tests #
#########

# Train an MLP network generator using VPG, PGB, and PPO.
@pytest.mark.parametrize('train_fn', [vpg_helper, pgb_helper, ppo_helper])
def test_generate_mlp(mnist_dataset, train_fn):
    train_fn(*mlp_helper(mnist_dataset))

# Train an CNN network generator using VPG, PGB, and PPO.
@pytest.mark.parametrize('train_fn', [vpg_helper, pgb_helper, ppo_helper])
def test_generate_cnn(mnist_dataset, train_fn):
    train_fn(*cnn_helper(mnist_dataset))

# Train an MLP+CNN network generator using VPG, PGB, and PPO.
@pytest.mark.parametrize('train_fn', [vpg_helper, pgb_helper, ppo_helper])
def test_generate_all(mnist_dataset, train_fn):
    train_fn(*joint_helper(mnist_dataset))

# Check that our extra statistics are logged correctly!
def test_extra_logging(mnist_dataset):
    # Logger that verifies extra info is present at each timestep.
    def local_logger(_, task_samples):
        for task in task_samples:
            for trajectory in task.trajectories:
                assert trajectory.enable_extra
                assert len(trajectory.extra) > 0
                for timestep in trajectory.extra:
                    l = trajectory.extra[timestep]
                    # Either we have accuracy / params, or the config was flagged as broken.
                    assert 'accuracy' in l or 'broken' in l 

    hypers, env, _, actor = cnn_helper(mnist_dataset)
    agent = librl.agent.pg.REINFORCEAgent(actor)
    agent.train()
    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=1, 
        replay_ctor=nasrl.replay.ProductEpisodeWithExtraLogs))
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step, log_fn=local_logger)