import unittest
import functools

import librl.agent.pg
import librl.nn.core, librl.nn.actor
import librl.reward
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc
import librl.utils
import torch, torch.utils
import torchvision.datasets, torchvision.transforms

import nasrl.task
import nasrl.tree.actor, nasrl.tree.env
import nasrl.tree.train

# Integration tests that demonstrates generating MLP's
# to solve the mnist classification using tree-based policies.
class MNISTClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hypers = {}
        hypers['device'] = 'cpu'
        hypers['epochs'] = 1
        hypers['task_count'] = 1
        cls.hypers = hypers

    def test_generate_mlp(self):
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),]) 
        train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
        validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)
        # Construct dataloaders from datasets
        t_loaders, v_loaders = librl.utils.load_split_data(train_dset, 100, 3), librl.utils.load_split_data(validation_dset, 1000, 1)
        env = nasrl.tree.env.MLPClassificationEnv((1,28,28), 10, torch.nn.CrossEntropyLoss(), t_loaders, v_loaders)
        # Construct my agent.
        x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
        policy_kernel = librl.nn.core.MLPKernel(x)
        policy_net = nasrl.tree.actor.MLPTreeActor(policy_kernel, env.observation_space)
        agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
        agent.train()
        # Construct meta-task
        dist = librl.task.TaskDistribution()
        dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=2, 
            sample_fn = nasrl.tree.train.sample_trajectories))
        nasrl.tree.train.cc_episodic_trainer(self.hypers, dist, librl.train.cc.policy_gradient_step)

    def test_generate_cnn(self):
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),]) 
        train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
        validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)
        # Construct dataloaders from datasets
        t_loaders, v_loaders = librl.utils.load_split_data(train_dset, 100, 3), librl.utils.load_split_data(validation_dset, 1000, 1)
        env = nasrl.tree.env.CNNClassificationEnv((1,28,28), 10, torch.nn.CrossEntropyLoss(), t_loaders, v_loaders)
        # Construct my agent.
        x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
        policy_kernel = librl.nn.core.MLPKernel(x)
        policy_net = nasrl.tree.actor.CNNTreeActor(policy_kernel, env.observation_space)
        agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
        agent.train()
        # Construct meta-task
        dist = librl.task.TaskDistribution()
        dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=3, 
            sample_fn = nasrl.tree.train.sample_trajectories))
        nasrl.tree.train.cc_episodic_trainer(self.hypers, dist, librl.train.cc.policy_gradient_step)

if __name__ == "__main__":
    unittest.main()