import cloudpickle
import nasrl
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt



plt.figure(figsize=(10,5))
# plt2.figure(figsize=(10,5))

# 		(manually change this)
# |
# |		list of paths to logs
# |		plot accuracy and network size per epoch
# |		make sure each path ends with a  '/'
# |
# V
log_directories = [
	'../../mlp_sweep_logs/sweep-pgb-mlp/', 
	'../../mlp_sweep_logs/sweep-vpg-mlp/',
	'../../mlp_sweep_logs/sweep-ppo-mlp/'
	]
# log_directories = ['../../mlp_sweep_logs/sweep-pgb-mlp/']

log_names = [
	'MLP_PGB',
	'MLP_VPG',
	'MLP_PPO'
	]


max_accuracies=[]
max_params=[]
all_params=[]
step=None
for lidx,log_directory in enumerate(log_directories):
	# stats from this epoch
	epoch_stats=[]
	# for epoch in logs
	epoch_dirs = os.listdir(log_directory)
	if lidx==0:
		epoch_dirs.sort()
	for eidx, epoch_dir in enumerate(epoch_dirs):
		# print('e_{} ({})'.format(idx,epoch_dir))
		# every element in task stats is a dict for that task
		task_stats={'accuracy' : [],
					'params' : []
					}
		# for task in epoch
		task_dirs = os.listdir(log_directory+epoch_dir)
		task_dirs.sort()
		for tidx,task in enumerate(task_dirs):
			# print('\t{} {}'.format(tidx,task))
			# only one trajetory per task
			trajectories = os.listdir(log_directory + epoch_dir + '/' + task)
			# f = open(log_directory + epoch_dir + '/' + task + '/'+traj_0, 'rb')
			# task_trajectory = cloudpickle.load(f)
			# f.close()
			# trajectories = task_obj.trajectories
			# rewards = trajectories.reward_buffer
			# print("\t\ttrajectories : ",len(task_obj.trajectories))
			# print("\t\tnum_steps : ", (task_obj._episode_length))
			task_dict={'accuracy': [],
						'params': []
						}
			# _episode_length = len(task_trajectory.reward_buffer)
			# _num_trajectories = 1

			for traj in trajectories:
				f = open(log_directory + epoch_dir + '/' + task + '/'+traj, 'rb')
				ep = cloudpickle.load(f)
				f.close()
				# ep = task_trajectory[i]
				rewards = ep.reward_buffer
				_episode_length = len(ep.reward_buffer)
				# print("\t\t\t",i,"rewards : ", rewards.tolist())
				# print("\t\t\t",i,"extra : ", ep.extra)
				# stats from final step
				# task_dict['accuracy'].append(ep.extra[task_obj._episode_length-1]['accuracy'])
				# task_dict['params'].append(ep.extra[task_obj._episode_length-1]['params'])

				# episode_accuracy=[]
				# episode_params=[]
				# for step in range(task_obj._episode_length):
				# 	# print(ep.extra[step]['params'])
				# 	# print(ep.extra[step]['accuracy'][0].item())
				# 	episode_accuracy.append(ep.extra[step]['accuracy'][0].item())
				# 	episode_params.append(ep.extra[step]['params'])

				# append the final step stats
				task_dict['accuracy'].append(ep.extra[_episode_length-1]['accuracy'])
				task_dict['params'].append(ep.extra[_episode_length-1]['params'])

			# 	# append average of all episodes
			# 	# task_dict['accuracy'].append(np.mean(episode_accuracy))
			# 	# task_dict['params'].append(np.mean(episode_accuracy))

			# now task_dict has the acc/params from the last step of each episode
			# task_dict['accuracy'] = [acc_final_step1,acc_final_step2,...]
			task_dict['accuracy'].append(ep.extra[_episode_length-1]['accuracy'])
			task_dict['params'].append(ep.extra[_episode_length-1]['params'])

			task_stats['accuracy'].append(np.mean(task_dict['accuracy']))
			task_stats['params'].append(np.mean(task_dict['params']))

		# task_stats now has a task_dict for each task

		# get average accuracy and params from the last episode of each epoch
		avg_acc = np.mean(task_stats['accuracy'])
		avg_params = np.mean(task_stats['params'])
		# for tdict in task_stats:
		# 	print(tdict['accuracy'])
		# 	print(tdict['params'])
		# 	avg_acc += tdict['accuracy'][-1]
		# 	avg_params += tdict['params'][-1]

		#   divide total over num_tasks
		# avg_acc /= float(len(task_stats))
		# avg_params /= float(len(task_stats))

		# add averages over all tasks over all trajectories to epoch_stats
		epoch_stats.append({'accuracy':avg_acc,'params':avg_params})

	# plt.xlim(0,len(results[0]))
	# print('epoch stats\n____________')
	# for epstat in epoch_stats:
	# 	print('last acc: {}, last par: {}'.format(epstat['accuracy'],epstat['params']))
	# rewards=[]

	# step=[]
	# reward=[]
	# beta=[]
	step = [i for i in range(len(epoch_dirs))]
	accuracy = [epoch['accuracy'] for epoch in epoch_stats]
	params = [epoch['params'] for epoch in epoch_stats]
	all_params.append(params)
	# print(step)
	# print(accuracy)
	# print(params)
	# reward = results[1][1:]
	# beta = results[2][1:]
	# print(step[0])
	# print(reward[0])
	# print(min(reward))

	# plot this log
	plt.plot(step,accuracy,label=log_names[lidx],linewidth=.5)
	max_accuracies.append(np.max(accuracy))
	max_params.append(np.max(params))


plt.ylim(0,np.max(max_accuracies) + 0.1)
plt.xlim(0,len(step)-1)
plt.legend(loc="best")
plt.xlabel("epoch")
plt.ylabel("end of training accuracy")
savefigname ='../../mlp_sweep_accuracy.pdf'
plt.savefig(savefigname)
# print('time to exit')
# sys.exit()

plt.clf()
plt.figure(figsize=(8,6))
plt.xlabel("epoch")
plt.ylabel("end of training network size")

for pidx,params in enumerate(all_params):
	step = [i for i in range(len(params))]
	plt.plot(step,params,label=log_names[pidx],linewidth=.5)

plt.ylim(0,np.max(max_params)*1.1)
plt.xlim(0,len(step)-1)
plt.legend(loc="best")
savefigname ='../../mlp_sweep_params.pdf'
plt.savefig(savefigname)