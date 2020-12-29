import argparse
import enum
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
        subdir = os.path.join(self.log_dir, f"epoch-{epochs}")
        # I would be very concerned if the subdir already exists
        os.makedirs(subdir)
        accuracy_list = []
        for task_idx, task in enumerate(task_samples):
            task_subdir = os.path.join(subdir, f"task-{task_idx}")
            os.makedirs(task_subdir)
            for trajectory_idx, trajectory in enumerate(task.trajectories):
                with open(os.path.join(task_subdir, f"traj{trajectory_idx}.pkl"), "wb") as fptr:
                    pickle.dump(trajectory, fptr)
                    accuracy_list.append(trajectory.extra[len(trajectory.extra)-1]['accuracy'][-1])
        print(f"Average accuracy for epoch {epochs} was {sum(accuracy_list)/len(accuracy_list)}.")


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

def get_mlp_conf(max_layers):
    interval = nasrl.field.Interval(0, 1000)
    field = nasrl.field.Field(interval, lambda *_:200)
    return nasrl.field.MLPConf(max_layers, field , min_layer_size=10)
    
def get_cnn_conf(max_layers):
    from nasrl.field.field import Interval, Field
    from nasrl.field.conf import CNNConf
    # Choose sensible ranges for (c)hannels, (k)ernel size, (s)tride,
    # (p)adding, and (d)ilation. `lambda` repesent sensible initial values.
    c_i, k_i = Interval(1, 128), Interval(2, 8)
    s_i, p_i, d_i = Interval(1, 4), Interval(0, 4), Interval(1, 4)
    c_f, k_f = Field(c_i, lambda *_:4), Field(k_i, lambda *_:4)
    s_f, p_f, d_f = Field(s_i, lambda *_:1), Field(p_i, lambda *_:2), Field(d_i, lambda *_:1)
    return CNNConf(max_layers, c_f, k_f, s_f, p_f, d_f, lambda _: 1)

def build_mlp(dset, hypers):
    t,v = dset.t_loaders, dset.v_loaders
    env = nasrl.tree.env.MLPClassificationEnv(dset.dims, get_mlp_conf(hypers['max_mlp_layers']),
        inner_loss = torch.nn.CrossEntropyLoss(),  adapt_steps=hypers['adapt_steps'],
        train_data_iter=t, validation_data_iter=v, classes=10, reward_fn=nasrl.reward.Linear, device='cpu')
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.RecurrentKernel(x, 100, 3)
    tx_layer_size = env.mlp_conf.mlp.transform.forward(env.mlp_conf.min_layer_size)
    policy_net = nasrl.tree.actor.MLPTreeActor(policy_kernel, env.observation_space, min_layer_size=tx_layer_size)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net = librl.nn.critic.ValueCritic(critic_kernel)
    return env, critic_net, policy_net

def build_cnn(dset, hypers):
    t,v = dset.t_loaders, dset.v_loaders
    env = nasrl.tree.env.CNNClassificationEnv(dset.dims, get_cnn_conf(hypers['max_cnn_layers']), 
        inner_loss = torch.nn.CrossEntropyLoss(),  adapt_steps=hypers['adapt_steps'],
        train_data_iter=t, validation_data_iter=v, classes=10, reward_fn=nasrl.reward.Linear, device='cpu')
    # Construct my agent.
    x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
    policy_kernel = librl.nn.core.RecurrentKernel(x, 100, 3)
    policy_net = nasrl.tree.actor.CNNTreeActor(policy_kernel, env.observation_space)
    critic_kernel = librl.nn.core.MLPKernel(x)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return env, critic_net, policy_net

def build_joint(dset, hypers):
    t, v = dset.t_loaders, dset.v_loaders
    env = nasrl.tree.env.JointClassificationEnv(dset.dims, get_cnn_conf(hypers['max_cnn_layers']), 
        get_mlp_conf(hypers['max_mlp_layers']), inner_loss = torch.nn.CrossEntropyLoss(),  adapt_steps=hypers['adapt_steps'],
        train_data_iter=t, validation_data_iter=v, classes=10, reward_fn=nasrl.reward.Linear,device='cpu' ) 
        # Construct an NN to process MLP and CNN network descriptions.
    cnn_size = functools.reduce(lambda x,y: x*y, env.cnn_observation_space.shape, 1)
    mlp_size = functools.reduce(lambda x,y: x*y, env.mlp_observation_space.shape, 1)
    cnn_policy_kernel = librl.nn.core.RecurrentKernel(cnn_size, 100, 3)
    mlp_policy_kernel = librl.nn.core.RecurrentKernel(mlp_size, 100, 3)

    # Use a bi-linear layer to combine state information about the MLP and CNN
    # to properly init cnn/mlp weighs.
    fusion_kernel = librl.nn.core.BilinearKernel(cnn_policy_kernel, mlp_policy_kernel, 10)
    tx_layer_size = env.mlp_conf.mlp.transform.forward(env.mlp_conf.min_layer_size)
    policy_net = nasrl.tree.actor.JointTreeActor(cnn_policy_kernel, mlp_policy_kernel, fusion_kernel, env.observation_space, min_layer_size=tx_layer_size)
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
    hypers['epochs'] = args.epochs
    hypers['task_count'] = args.task_count
    hypers['adapt_steps'] = args.adapt_steps
    hypers['episode_length'] = args.episode_length
    hypers['max_mlp_layers'] = 10
    hypers['max_cnn_layers'] = 10
    dset = mnist_dataset()

    env, critic, actor = args.type(dset, hypers)
    critic, actor = critic.to(hypers['device']), actor.to(hypers['device'])
    agent = args.alg(hypers, critic, actor)

    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=args.adapt_steps, 
        replay_ctor=nasrl.replay.ProductEpisodeWithExtraLogs))
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step, log_fn=DirLogger(args.log_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generator network on a task for multiple epochs and record results.")
    parser.add_argument("--log-dir", dest="log_dir", help="Directory in which to store logs.")
    # Choose RL algorthim.
    learn_alg_group = parser.add_mutually_exclusive_group()
    learn_alg_group.add_argument("--vpg", action='store_const', const=vpg_helper, dest='alg', help="Train agent using VPG.")
    learn_alg_group.add_argument("--pgb", action='store_const', const=pgb_helper, dest='alg', help="Train agent using PGB.")
    learn_alg_group.add_argument("--ppo", action='store_const', const=ppo_helper, dest='alg', help="Train agent using PPO.")
    learn_alg_group.set_defaults(alg=vpg_helper)
    # Choose which kind of network to generate.
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument("--mlp", action='store_const', const=build_mlp, dest='type', help="Build MLP only networks.")
    type_group.add_argument("--conv", action='store_const', const=build_cnn, dest='type', help="Build CNN only networks.")
    type_group.add_argument("--joint", action='store_const', const=build_joint, dest='type', help="Build mixed CNN+MLP networks.")
    type_group.set_defaults(type=build_mlp)
    # Task distribution hyperparams.
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for which to train the generator network.")
    parser.add_argument("--adapt-steps", dest="adapt_steps", default=2, type=int, help="Number of epochs which to the generated network.")
    parser.add_argument("--task-count", dest="task_count", default=3, type=int, help="Number of times of trials of the generator per epoch.")
    args = parser.parse_args()
    main(args)

@pytest.mark.parametrize("alg", ["vpg", "pgb"])
@pytest.mark.parametrize("type", ["mlp","cnn", "joint"])
def test_all(alg, type):
    from types import SimpleNamespace

    args = {}
    args['epochs'] = 100
    args['task_count'] = 1
    args['episode_length'] = 100
    args['adapt_steps'] = 3

    if alg == "vpg": args['alg'] = vpg_helper
    elif alg == "pgb": args['alg'] = pgb_helper
    elif alg == "ppo": args['alg'] = ppo_helper

    if type == "mlp": args['type'] = build_mlp
    elif type == "cnn": args['type'] = build_cnn
    elif type == "joint": args['type'] = build_joint
    
    args['log_dir'] = f"test-LSTM-{alg}-{type}"
    
    return main(SimpleNamespace(**args))