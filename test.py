import os
import sys

import gym
import time
import numpy as np
import gym_novel_gridworlds

# sys.path.append('gym_novel_gridworlds/envs')
# from novel_gridworld_v0_env import NovelGridworldV0Env
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt



def CheckTrainingDoneCallback(reward_array, done_array, env):

	done_cond = False
	reward_cond = False
	if len(done_array) > 30:
		if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
			if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
				done_cond = True

		if done_cond == True:
			if env < 3:
				if np.mean(reward_array[-10:]) > 950:
					reward_cond = True
			else:
				if np.mean(reward_array[-10:]) > 950:
					reward_cond = True

		if done_cond == True and reward_cond == True:
			return 1
		else:
			return 0
	else:
		return 0



if __name__ == "__main__":

	no_of_environmets = 4

	width_array = [8,11,13,13]
	height_array = [8,11,13,13]
	cube1_array = [1,0,1,1]
	cube2_array = [0,1,0,1]
	crafting_table_array = [0,0,1,1]
	type_of_env_array = [0,1,2,2]

	total_timesteps_array = []
	total_reward_array = []
	avg_reward_array = []
	final_timesteps_array = []
	final_reward_array = []
	final_avg_reward_array = []
	curr_task_completion_array = []
	final_task_completion_array = []

	actionCnt = 6
	D = 33 #8 beams x 4 items lidar + 1 inventory items
	NUM_HIDDEN = 10
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1
	random_seed = 1

	# agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	# agent.set_explore_epsilon(MAX_EPSILON)
	action_space = ['W','A','D','U','C']
	total_episodes_arr = []

	for i in range(no_of_environmets):
		# print("Environment: ", i)
		# i = 2

		width = width_array[i]
		height = height_array[i]
		no_cube1 = cube1_array[i]
		no_cube2 = cube2_array[i]
		crafting_table = crafting_table_array[i]
		type_of_env = type_of_env_array[i]

		final_status = False

		if i == 0:
			agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
			agent.set_explore_epsilon(MAX_EPSILON)
			agent.load_model(0,0,i)
		else:
			agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
			agent.set_explore_epsilon(MAX_EPSILON)
			agent.load_model(0,0,i-1)
			agent.reset()
			print("loaded model")


		if i == no_of_environmets-1:
			final_status = True

		env_id = 'NovelGridworld-v0'
		env = gym.make(env_id, map_width = width, map_height = height, items_quantity = {'cube1': no_cube1, 'cube2': no_cube2, 'crafting_table': crafting_table}, goal_env = type_of_env, is_final = final_status)
		
		t_step = 0
		episode = 0
		t_limit = 100
		reward_sum = 0
		reward_arr = []
		avg_reward = []
		done_arr = []
		env_flag = 0

		env.render()
		env.reset()

		while True:
			
			# get obseration from sensor
			env.render()
			obs = env.get_observation()
		
			# act 
			a = agent.process_step(obs,True)
			
			new_obs, reward, done, info = env.step(a)

			# give reward
			agent.give_reward(reward)
			reward_sum += reward
			
			t_step += 1
			
			if t_step > t_limit or done == True:
				
				# finish agent
				if done == True:
					done_arr.append(1)
					curr_task_completion_array.append(1)
				elif t_step > t_limit:
					done_arr.append(0)
					curr_task_completion_array.append(0)
				
				print("\n\nfinished episode = "+str(episode)+" with " +str(reward_sum)+"\n")

				reward_arr.append(reward_sum)
				avg_reward.append(np.mean(reward_arr[-40:]))

				total_reward_array.append(reward_sum)
				avg_reward_array.append(np.mean(reward_arr[-40:]))
				total_timesteps_array.append(t_step)
		
				done = True
				t_step = 0
				agent.finish_episode()
			
				# update after every episode
				if episode % 10 == 0:
					agent.update_parameters()
			
				# reset environment
				episode += 1

				env.reset()
				env.render()
				reward_sum = 0


				env_flag = 0
				# if i< 3:
				# 	env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)
		
				# quit after some number of episodes
				if episode > 120000 or env_flag == 1:

					agent.save_model(0,0,i)
					total_episodes_arr.append(episode)

					break
