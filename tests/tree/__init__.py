
import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.reward, librl.replay.episodic
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc

######################
# Execute train loop #
######################
def vpg_helper(hypers, env, _, policy_net):
    agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent.train()
    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=1, 
        replay_ctor=librl.replay.episodic.ProductEpisode))
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step)

def pgb_helper(hypers, env, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PGB(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=1, 
        replay_ctor=librl.replay.episodic.ProductEpisode))
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step)

def ppo_helper(hypers, env, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PPO(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    dist = librl.task.TaskDistribution()
    dist.add_task(librl.task.Task.Definition(librl.task.ContinuousGymTask, env=env, agent=agent, episode_length=1, 
        replay_ctor=librl.replay.episodic.ProductEpisode))
    librl.train.train_loop.cc_episodic_trainer(hypers, dist, librl.train.cc.policy_gradient_step)