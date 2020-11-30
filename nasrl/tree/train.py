import torch
import numpy as np

import librl.utils

# Re-implement episodic trainer that does not collect action statistics.
# As soon as logging is done via a log_fn in librl, this overload can be dropped
def cc_episodic_trainer(train_info, task_dist, train_fn):
    for epoch in range(train_info['epochs']):
        task_samples = task_dist.sample(train_info['task_count'])
        train_fn(task_samples)
        rewards = len(task_samples) * [None]
        for idx, task in enumerate(task_samples): 
            rewards[idx] = sum(task.trajectories[0].reward_buffer.view(-1))
            print(task.trajectories[0].logprob_buffer)

        mean_reward = (sum(rewards)/len(rewards)).item() # type: ignore
        print(f"R^bar_({epoch}) = {mean_reward}.")
