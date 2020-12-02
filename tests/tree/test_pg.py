import functools

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import pytest
import torch, torch.utils

import nasrl.nn
import nasrl.tree.actor, nasrl.tree.env

from . import * 
#######################
# Agent / environment #
#######################
def mlp_helper(mnist_dataset):
    loaders = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    env = nasrl.tree.env.MLPClassificationEnv((1,28,28), 10, torch.nn.CrossEntropyLoss(), *loaders)
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = nasrl.tree.actor.MLPTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return mnist_dataset.hypers, env, critic_net, policy_net

def cnn_helper(mnist_dataset):
    loaders = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    env = nasrl.tree.env.CNNClassificationEnv((1,28,28), 10, torch.nn.CrossEntropyLoss(), *loaders)
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = nasrl.tree.actor.CNNTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return mnist_dataset.hypers, env, critic_net, policy_net

def joint_helper(mnist_dataset):
    loaders = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    env = nasrl.tree.env.JointClassificationEnv((1,28,28), 10, 10, torch.nn.CrossEntropyLoss(), *loaders)
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
    train_fn(*mlp_helper(mnist_dataset))