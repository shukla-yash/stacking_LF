import numpy as np
import matplotlib.pyplot as plt
import os


num_random_seeds = 1
seed_arr = [1]
log_dir = 'logs'
data_total_timesteps = [[] for _ in range(num_random_seeds)]
data_total_reward = [[] for _ in range(num_random_seeds)]
data_avg_reward = [[] for _ in range(num_random_seeds)]
data_total_episodes = [[] for _ in range(num_random_seeds)]
data_final_timesteps = [[] for _ in range(num_random_seeds)]
data_final_reward = [[] for _ in range(num_random_seeds)]
data_final_avg_reward = [[] for _ in range(num_random_seeds)]

for i in range(num_random_seeds):
	log_dir = 'logs_' + str(seed_arr[i])
	random_seed = seed_arr[i]
	experiment_file_name_total_timesteps = 'randomseed_' + str(random_seed) + '_total_timesteps'
	path_to_save_total_timesteps = log_dir + os.sep + experiment_file_name_total_timesteps + '.npz'
	temp = np.load(path_to_save_total_timesteps)
	print("size: ", temp['curriculum_timesteps'].shape)
	data_total_timesteps[i] = temp['curriculum_timesteps']

	experiment_file_name_total_reward = 'randomseed_' + str(random_seed) + '_total_reward'
	path_to_save_total_reward = log_dir + os.sep + experiment_file_name_total_reward + '.npz'
	temp = np.load(path_to_save_total_reward)
	data_total_reward[i] = temp['curriculum_reward']
	print("length data total reward: ", len(data_total_reward[i]))

	experiment_file_name_avg_reward = 'randomseed_' + str(random_seed) + '_avg_reward'
	path_to_save_avg_reward = log_dir + os.sep + experiment_file_name_avg_reward + '.npz'
	temp = np.load(path_to_save_avg_reward)
	data_avg_reward[i] = temp['curriculum_avg_reward']
	print("data_avg_reward: ", len(data_avg_reward[i]))


	experiment_file_name_total_episodes = 'randomseed_' + str(random_seed) + '_total_episodes'
	path_to_save_total_episodes = log_dir + os.sep + experiment_file_name_total_episodes + '.npz'
	temp = np.load(path_to_save_total_episodes)
	data_total_episodes[i] = temp['curriculum_episodes']
	print("data _total_episodes: ", data_total_episodes)


	experiment_file_name_final_timesteps = 'randomseed_' + str(random_seed) + '_final_timesteps'
	path_to_save_final_timesteps = log_dir + os.sep + experiment_file_name_final_timesteps + '.npz'
	temp = np.load(path_to_save_final_timesteps)
	data_final_timesteps[i] = temp['final_timesteps']


	experiment_file_name_final_reward = 'randomseed_' + str(random_seed) + '_final_reward'
	path_to_save_final_reward = log_dir + os.sep + experiment_file_name_final_reward + '.npz'
	temp = np.load(path_to_save_final_reward)
	data_final_reward[i] = temp['final_reward']


	experiment_file_name_final_avg_reward = 'randomseed_' + str(random_seed) + '_final_avg_reward'
	path_to_save_final_avg_reward = log_dir + os.sep + experiment_file_name_final_avg_reward + '.npz'
	temp = np.load(path_to_save_final_avg_reward)
	data_final_avg_reward[i] = temp['final_avg_reward']


total_reward_arr_shifted = [[] for _ in range(num_random_seeds)]
avg_reward_arr_shifted = [[] for _ in range(num_random_seeds)]


final_reward_shifted = [[] for _ in range(num_random_seeds)]
curriculum_episodes = []
for i in range(num_random_seeds):
	curriculum_episodes.append(sum(data_total_episodes[i][:4]))
	total_reward_arr_shifted[i] = [np.nan] * curriculum_episodes[i]
	print("curriculum_eps: ", curriculum_episodes)
	print("length before: ", len(total_reward_arr_shifted[i]))
	print("extending length: ", len(data_total_reward[i][curriculum_episodes[i]:]))
	total_reward_arr_shifted[i].extend(data_total_reward[i][curriculum_episodes[i]:])
	print("length after: ", len(total_reward_arr_shifted[i]))
	avg_reward_arr_shifted[i] = [np.nan] * curriculum_episodes[i]
	avg_reward_arr_shifted[i].extend(data_avg_reward[i][curriculum_episodes[i]:])
	print("reward thing: ", len(avg_reward_arr_shifted[i]))

print("hereee00")

curriculum_shifted_reward_arr = []

max_len = max(len(avg_reward_arr_shifted[i]) for i in range(len(avg_reward_arr_shifted)))
print("maxlen is: ", max_len)


for i in range(num_random_seeds):
	print(len(avg_reward_arr_shifted[i]))
	if len(avg_reward_arr_shifted[i]) < max_len:
		while len(avg_reward_arr_shifted[i]) < max_len:
			avg_reward_arr_shifted[i].append(np.nan)
# print("len 0:", len(avg_reward_arr_shifted[0]))
# print("len 1:", len(avg_reward_arr_shifted[1]))


for i in range(max_len):
	count = 0
	sum_rewards = 0
	avg_rewards = 0
	for j in range(len(avg_reward_arr_shifted)):
		if not np.isnan(avg_reward_arr_shifted[j][i]):
			count += 1 
			sum_rewards += avg_reward_arr_shifted[j][i]
			avg_rewards = sum_rewards/count
		if j == len(avg_reward_arr_shifted)-1 and count == 0:
			curriculum_shifted_reward_arr.append(np.nan)
		elif j == len(avg_reward_arr_shifted) - 1:
			curriculum_shifted_reward_arr.append(avg_rewards)


print("data data_final_average:", data_final_avg_reward[0])
# print("data data_final_average:", len(data_final_avg_reward[1]))

data_final_average = []

for i in range(len(data_final_avg_reward[0])):
	avg = 0
	sum_data_final = 0
	for j in range(len(data_final_avg_reward)):
		if data_final_avg_reward[j][i] > 1000:
			data_final_avg_reward[j][i] = 0
			sum_data_final += data_final_avg_reward[j][i]
		else:
			sum_data_final += data_final_avg_reward[j][i]

	# if data_final_avg_reward[0][i] > 1000:
	# 	data_final_avg_reward[0][i] = 0

	# if data_final_avg_reward[1][i] > 1000:
	# 	data_final_avg_reward[1][i] = 0
	avg = sum_data_final/num_random_seeds
	data_final_average.append(avg)

for i in range(len(curriculum_shifted_reward_arr)):
	if curriculum_shifted_reward_arr[i] > 750:
		curriculum_shifted_reward_arr[i] = 750

max_latest_curr = np.max(curriculum_episodes)

for i in range(max_latest_curr):
	curriculum_shifted_reward_arr[i] = np.nan

print("len final:", len(curriculum_shifted_reward_arr))
plt.plot(curriculum_shifted_reward_arr, label = 'learning through curriculum')
plt.plot(data_final_average, label = 'learning from scratch')
plt.xlabel('Episodes')
plt.ylabel('Average rewards')
plt.legend()
# plt.show()
plt.savefig("LF_HC_fire")
print("doing everything")
