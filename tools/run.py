import argparse
import functools
import pickle
import os

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import pytest
import torch, torch.utils

import nasrl.nn
import nasrl.tree.actor, nasrl.tree.env
import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.reward, librl.replay.episodic
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc

import nasrl.replay

class DirLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir)

    def __call__(self, epochs, task_samples):
        subdir = os.path.join(self.log_dir, f"{epochs}")
        # I would be very concerned if the subdir already exists
        os.makedirs(subdir)
        for idx, task in enumerate(task_samples):
            with open(os.path.join(subdir, f"task{idx}.pkl"), "wb") as fptr:
                pickle.dump(task, fptr)
######################
#      Datasets      #
######################
def mnist_dataset():
    import torchvision.datasets, torchvision.transforms
    from librl.utils import load_split_data
    class helper:
        def __init__(self):
            transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.1307, 0.3081)]) 
            # Construct dataloaders from datasets
            train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
            validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)
            self.t_loaders, self.v_loaders = load_split_data(train_dset, 100, 3), load_split_data(validation_dset, 1000, 1)
            self.dims = (1, 28, 28)
    return helper()

def cifar10_dataset():
    import torchvision.datasets, torchvision.transforms
    from librl.utils import load_split_data
    class helper:
        def __init__(self):
            mean, stdev = .5, .25
            transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((mean, mean, mean), (stdev, stdev, stdev))]) 

            # Construct dataloaders from datasets
            train_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, download=True)
            validation_dset = torchvision.datasets.CIFAR10("__pycache__/CIFAR10", transform=transformation, train=False)
            self.t_loaders, self.v_loaders = load_split_data(train_dset, 100, 3), load_split_data(validation_dset, 1000, 1)
            self.dims = (3, 32, 32)
    return helper()


######################
# Execute train loop #
######################
def vpg_helper(hypers, _, policy_net):
    agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent.train()
    return agent
    

def pgb_helper(hypers, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PGB(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

def ppo_helper(hypers, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PPO(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

#######################
# Agent / environment #
#######################
def build_mlp(dset, hypers):
    loaders = dset.t_loaders, dset.v_loaders
    env = nasrl.tree.env.MLPClassificationEnv(dset.dims, hypers['max_mlp_layers'], torch.nn.CrossEntropyLoss(), *loaders, adapt_steps=0)
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = nasrl.tree.actor.MLPTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net = librl.nn.critic.ValueCritic(critic_kernel)
    return env, critic_net, policy_net

def build_cnn(dset, hypers):
    loaders = dset.t_loaders, dset.v_loaders
    env = nasrl.tree.env.CNNClassificationEnv(dset.dims, hypers['max_cnn_layers'], torch.nn.CrossEntropyLoss(), *loaders, adapt_steps=0)
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.MLPKernel(x)
    policy_net = nasrl.tree.actor.CNNTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return env, critic_net, policy_net

def build_joint(dset, hypers):
    loaders = dset.t_loaders, dset.v_loaders
    env = nasrl.tree.env.JointClassificationEnv(dset.dims, hypers['max_cnn_layers'], hypers['max_mlp_layers'],
    torch.nn.CrossEntropyLoss(), *loaders)
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
    return env, critic_net, policy_net 

#######################
#     Entry point     #
#######################
def main(args):
    hypers = {}
    hypers['device'] = 'cpu'
    hypers['epochs'] = 10
    hypers['task_count'] = 2
    hypers['episode_length'] = 2
    hypers['max_mlp_layers'] = 10
    hypers['max_cnn_layers'] = 10
    dset = mnist_dataset()

    env, critic, actor = build_mlp(dset, hypers)
    agent = vpg_helper(hypers, critic, actor)

    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=hypers['episode_length'], 
        replay_ctor=nasrl.replay.ProductEpisodeWithExtraLogs))
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step, log_fn=DirLogger("./l0gs"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do things")
    args = parser.parse_args()
    main(args)