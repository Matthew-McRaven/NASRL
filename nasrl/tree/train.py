
import torch
import numpy as np

import librl.utils


# Enapsulate all replay memory of a single task
# Acts like librl.replay.episodic.Episode, except the action is a numpy array of objects rather
# than a tensor of actual values.
class WeirdActionEpisode:
    def __init__(self, obs_space, episode_length=200, device='cpu'):
        self.state_buffer = torch.zeros([episode_length, *obs_space.shape], dtype=librl.utils.convert_np_torch(obs_space.dtype)).to(device) # type: ignore
        self.action_buffer = np.full([episode_length], None, dtype=object)# type: ignore
        self.logprob_buffer = torch.zeros([episode_length], dtype=torch.float32).to(device) # type: ignore
        self.reward_buffer = torch.zeros([episode_length], dtype=torch.float32).to(device) # type: ignore
        self.policy_buffer = np.full([episode_length], None, dtype=object)
        self.done =  None

    def log_state(self, t, state):
        self.state_buffer[t] = state
    def log_action(self, t, action, logprob):
        self.action_buffer[t] = action
        self.logprob_buffer[t] = logprob
    def log_rewards(self, t, reward):
        self.reward_buffer[t] = reward
    def log_policy(self, t, policy):
        self.policy_buffer[t]= policy
    def log_done(self, t):
        self.done = t

    def clear_replay(self):
        map(lambda x: x.fill_(0).detach_(), [self.state_buffer, self.action_buffer, self.logprob_buffer, self.reward_buffer])
        self.policy_buffer.fill(None)
        self.done = None

# Must re-implement trajectory sampler to not convert action to a numpy array.
# Actions returned by our actor are lists by default, and thus don't need to be converted from tensors.
def sample_trajectories(task):
    task.clear_trajectories()
    task.init_env()
    for i in range(task.trajectory_count):
        state = torch.tensor(task.env.reset()).to(task.device) # type: ignore
        episode = WeirdActionEpisode(task.env.observation_space, task.episode_length)
        episode.log_done(task.episode_length + 1)
        for t in range(task.episode_length):
            
            episode.log_state(t, state)

            action, logprob_action = task.agent.act(state)
            #print(action)
            episode.log_action(t, action, logprob_action)
            if task.agent.policy_based: episode.log_policy(t, task.agent.policy_latest)
            state, reward, done, _ = task.env.step(action)
            if task.agent.allow_callback: task.agent.act_callback(state=state, reward=reward)
            if not torch.is_tensor(state): state = torch.tensor(state).to(task.device) # type: ignore
            if not torch.is_tensor(reward): reward = torch.tensor(reward).to(task.device) # type: ignore

            episode.log_rewards(t, reward)
            if done: 
                episode.log_done(t+1)
                break

        task.add_trajectory(episode)

# Re-implement episodic trainer that does not collect action statistics.
# As soon as logging is done via a log_fn in librl, this overload can be dropped
def cc_episodic_trainer(train_info, task_dist, train_fn):
    for epoch in range(train_info['epochs']):
        task_samples = task_dist.sample(train_info['task_count'])
        print(len(task_samples))
        train_fn(task_samples)
        rewards = len(task_samples) * [None]
        for idx, task in enumerate(task_samples): 
            rewards[idx] = sum(task.trajectories[0].reward_buffer.view(-1))
            print(task.trajectories[0].logprob_buffer)

        mean_reward = (sum(rewards)/len(rewards)).item() # type: ignore
        print(f"R^bar_({epoch}) = {mean_reward}.")
