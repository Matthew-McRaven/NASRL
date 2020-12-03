import functools

from librl import replay
import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.reward, librl.replay.episodic
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc
import librl.utils
import pytest
import torch, torch.utils

import nasrl.nn
import nasrl.task
import nasrl.tree.actor, nasrl.tree.env
import nasrl.reward
import nasrl.field

from . import *

def mlp_helper(mnist_dataset, reward_fn):
    loaders = mnist_dataset.t_loaders, mnist_dataset.v_loaders
    # TODO: If last adapt step stops updating NN, must change this to 1.
    interval = nasrl.field.Interval(0, 1000)
    field = nasrl.field.Field(interval, lambda *_:200)
    mlp_conf = nasrl.field.MLPConf(10, field, min_layer_size=10)
    env = nasrl.tree.env.MLPClassificationEnv((1,28,28), mlp_conf, torch.nn.CrossEntropyLoss(), *loaders, reward_fn=reward_fn, adapt_steps=0)
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = nasrl.tree.actor.MLPTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return mnist_dataset.hypers, env, critic_net, policy_net

# Try out our various reward functions.
@pytest.mark.parametrize('reward_fn', [nasrl.reward.Linear, nasrl.reward.PolyAsymptotic()])
def test_reward_fn(mnist_dataset, reward_fn):
    vpg_helper(*mlp_helper(mnist_dataset, reward_fn))

# Try out our various complexity penalties.
@pytest.mark.parametrize('penalty_fn', [nasrl.reward.SizePenalty, nasrl.reward.DepthPenalty])
def test_reward_fn(mnist_dataset, penalty_fn):
    vpg_helper(*mlp_helper(mnist_dataset, nasrl.reward.PolyAsymptotic(penalty_fn=penalty_fn)))