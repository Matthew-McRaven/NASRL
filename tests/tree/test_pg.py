import librl.agent.pg
import librl.task, librl.train.train_loop, librl.train.cc
import pytest

from . import * 

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