import unittest
import functools
from librl import replay

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.reward, librl.replay.episodic
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc
import librl.utils
import torch, torch.utils
import torchvision.datasets, torchvision.transforms

import nasrl.nn
import nasrl.task
import nasrl.tree.actor, nasrl.tree.env

# Integration tests that demonstrates generating MLP's, CNN's,
# and a mixture to solve the MNIST classification using tree-based policies.
class ActorCriticMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hypers = {}
        hypers['device'] = 'cpu'
        hypers['epochs'] = 1
        hypers['task_count'] = 1
        cls.hypers = hypers
        transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),]) 
        # Construct dataloaders from datasets
        train_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, download=True)
        validation_dset = torchvision.datasets.MNIST("__pycache__/MNIST", transform=transformation, train=False)
        cls.t_loaders, cls.v_loaders = librl.utils.load_split_data(train_dset, 100, 3), librl.utils.load_split_data(validation_dset, 1000, 1)
    
    def mlp_helper(self):
        env = nasrl.tree.env.MLPClassificationEnv((1,28,28), 10, torch.nn.CrossEntropyLoss(), self.t_loaders,self. v_loaders)
        # Construct my agent.
        x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
        policy_kernel = librl.nn.core.MLPKernel(x)
        policy_net = nasrl.tree.actor.MLPTreeActor(policy_kernel, env.observation_space)
        critic_kernel = librl.nn.core.MLPKernel(x)
        critic_net= librl.nn.critic.ValueCritic(critic_kernel)
        return env, critic_net, policy_net

   
    def test_generate_mlp_pgb(self):
        self.pgb_helper(*self.mlp_helper())

    def cnn_helper(self):
        env = nasrl.tree.env.CNNClassificationEnv((1,28,28), 10, torch.nn.CrossEntropyLoss(), self.t_loaders, self.v_loaders)
        # Construct my agent.
        x = functools.reduce(lambda x,y: x*y, env.observation_space.shape, 1)
        policy_kernel = librl.nn.core.MLPKernel(x)
        policy_net = nasrl.tree.actor.CNNTreeActor(policy_kernel, env.observation_space)
        critic_kernel = librl.nn.core.MLPKernel(x)
        critic_net= librl.nn.critic.ValueCritic(critic_kernel)
        return env, critic_net, policy_net

    
    def test_generate_cnn(self):
        self.pgb_helper(*self.mlp_helper())

    def joint_helper(self):
        env = nasrl.tree.env.JointClassificationEnv((1,28,28), 10, 10, torch.nn.CrossEntropyLoss(), self.t_loaders, self.v_loaders)
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
    
    def test_generate_all_pgb(self):
        self.pgb_helper(*self.joint_helper())
    
    
    def pgb_helper(self, env, critic_net, policy_net):
        actor_loss = librl.nn.pg_loss.PGB(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
        agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
        agent.train()
        dist = librl.task.TaskDistribution()
        dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=1, 
            replay_ctor=librl.replay.episodic.ProductEpisode))
        librl.train.train_loop.cc_episodic_trainer(self.hypers, dist, librl.train.cc.policy_gradient_step)

if __name__ == "__main__":
    unittest.main()