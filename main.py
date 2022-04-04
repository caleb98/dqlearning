import os
import time

import cv2
import gym
import torch.cuda
import numpy as np

import networks
from training import Agent
from training import EnvironmentInterface
from training import QValueApproximationMethod
from training import Trainer

# The directory in which agent models should be saved.
AGENTS_DIR = ".\\agents"


def main():
	# Create the agents directory if necessary
	if not os.path.isdir(AGENTS_DIR):
		os.mkdir(AGENTS_DIR)
	
	# Select and create the environment. Other OpenAI gym environments may be selected by changing the string here.
	environment = "LunarLander-v2"
	env = gym.make(environment)

	# Get the number of possible actions that may be selected.
	num_inputs = 1
	for i in env.observation_space.shape:
		num_inputs *= i
	num_actions = env.action_space.n

	# Function for generating new networks (used to create the agents for training)
	def network_generator():
		network = networks.LinearNetwork(num_inputs, num_actions)
		network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		return network
	
	# Matrix of parameters for testing in the experiment
	test_parameters_matrix = [
		# Approximation Method,              PER,   Learning Rate   Multi Q Networks
		[QValueApproximationMethod.STANDARD, False, 0.1, 			0],
		[QValueApproximationMethod.STANDARD, False, 0.01, 			0],
		[QValueApproximationMethod.STANDARD, False, 0.001, 			0],
		[QValueApproximationMethod.STANDARD, False, 0.0001, 		0],
		[QValueApproximationMethod.STANDARD, True,  0.1, 			0],
		[QValueApproximationMethod.STANDARD, True,  0.01, 			0],
		[QValueApproximationMethod.STANDARD, True,  0.001, 			0],
		[QValueApproximationMethod.STANDARD, True,  0.0001, 		0],

		[QValueApproximationMethod.DOUBLE_Q_LEARNING, False, 0.1,		0],
		[QValueApproximationMethod.DOUBLE_Q_LEARNING, False, 0.01,		0],
		[QValueApproximationMethod.DOUBLE_Q_LEARNING, False, 0.001,		0],
		[QValueApproximationMethod.DOUBLE_Q_LEARNING, False, 0.0001,	0],
		[QValueApproximationMethod.DOUBLE_Q_LEARNING, True, 0.1,		0],
		[QValueApproximationMethod.DOUBLE_Q_LEARNING, True, 0.01,		0],
		[QValueApproximationMethod.DOUBLE_Q_LEARNING, True, 0.001,		0],
		[QValueApproximationMethod.DOUBLE_Q_LEARNING, True, 0.0001,		0],

		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.1,	4],
		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.01,	4],
		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.001,	4],
		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.0001,	4],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.1,		4],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.01,	4],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.001,	4],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.0001,	4],

		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.1,	8],
		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.01,	8],
		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.001,	8],
		[QValueApproximationMethod.MULTI_Q_LEARNING, False, 0.0001,	8],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.1,		8],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.01,	8],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.001,	8],
		[QValueApproximationMethod.MULTI_Q_LEARNING, True, 0.0001,	8]
	]
	
	# Train and evaluate the models
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	models = []
	model_eval_data = np.zeros(len(test_parameters_matrix))
	train_current = 1
	train_total = len(test_parameters_matrix)
	for test_params in test_parameters_matrix:
		
		# Train the agent
		agent = train_agent(
			environment,
			network_generator,
			
			qvalue_approx_method=test_params[0],
			use_per=test_params[1],
			learning_rate=test_params[2],
			multi_qlearn_networks=test_params[3],
			
			target_update=500,  # 100 for cartpole, 500 for lunar lander
			
			render_frames=False,
			show_plots=False,
			episodes=250
		)
		
		print(f"Finished training {train_current} of {train_total}.")
		
		# Evaluate the model
		print("Evaluating model...")
		
		models.append(f"{test_params[0]}{test_params[3] if not test_params[3] == 0 else ''}"
						f"{'_PER' if test_params[1] else ''}_{test_params[2]}")
		
		episode_rewards = np.zeros(500)
		for episode in range(500):
			# Reset environment and reward tracker
			state = env.reset()
			current_episode_reward = 0
			
			# Loop until episode complete
			while True:
				action = agent.dqn(
					torch.tensor([state], dtype=torch.float, device=device)
				).max(1)[1].item()
				state, reward, done, _ = env.step(action)
				current_episode_reward += reward
				
				# When episode is finished, add its reward to the rewards array
				if done:
					episode_rewards[episode] = current_episode_reward
					break
		
		# Add this model's average episode reward value to the evaluation array
		model_eval_data[train_current - 1] = np.average(episode_rewards)
		
		train_current += 1
	
	# Save the evaluation data
	env_dir = os.path.join(
		AGENTS_DIR,
		environment
	)
	
	if not os.path.isdir(env_dir):
		os.mkdir(env_dir)
	
	np.savetxt(
		os.path.join(env_dir, "evaluation_data.csv"),
		model_eval_data.transpose(),
		fmt="%.5f",
		delimiter=",",
		header=",".join(models)
	)
	

def train_agent(environment_name, network_generator, qvalue_approx_method: QValueApproximationMethod, use_per: bool,
				multi_qlearn_networks: int = 8,	render_frames: bool = True, episodes: int = 250,
				epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: int = 10000,
				discount_factor: float = 0.99, learning_rate: float = 0.001, target_update: int = 500,
				train_batch_size: int = 128, replay_memory_size: int = 250000, show_plots: bool = True):
	"""
	
	:param environment_name: Environment to train an agent in
	:param network_generator: The function used to generate the network used by the agent
	:param qvalue_approx_method: Which q-value approximation method to use in the training process
	:param use_per: Whether or not to use PER during training
	:param multi_qlearn_networks: Number of networks to use if multi q-learning is enabled
	:param render_frames: Whether or not to render the environment on the screen (disabling will result in faster training)
	:param episodes: Number of episodes to train the agent for
	:param epsilon_start: Starting value for epsilon in epsilon-greedy action selection
	:param epsilon_end: Ending value for epsilon in epsilon-greedy action selection
	:param epsilon_decay: Over how many training steps to decay epsilon from epsilon_start to epsilon_end
	:param discount_factor: How strongly future rewards are considered
	:param learning_rate: How strongly to update the network based on training loss
	:param target_update: Training step interval used for updating the target network when double dqn is used
	:param train_batch_size: Number of transitions to sample from replay memory in each training step
	:param replay_memory_size: Maximum transitions to store in replay memory
	:param show_plots: Whether or not to display plots during the training process
	:return: the trained agent
	"""
	# Check the environment directory and make it if necessary
	env_dir = os.path.join(
		AGENTS_DIR,
		environment_name
	)
	
	if not os.path.isdir(env_dir):
		os.mkdir(env_dir)
	
	# Check the agent directory and make it if necessary
	agent_dir = os.path.join(
		env_dir,
		f"{qvalue_approx_method}"
		f"{multi_qlearn_networks if qvalue_approx_method == QValueApproximationMethod.MULTI_Q_LEARNING else ''}"
		f"{'_PER' if use_per else ''}"
		f"_e{episodes}_epsb{epsilon_start}_epse{epsilon_end}_epsd{epsilon_decay}_df{discount_factor}"
		f"_lr{learning_rate}_tu{target_update}_tbs{train_batch_size}_rms{replay_memory_size}"
	)
	
	if not os.path.isdir(agent_dir):
		os.mkdir(agent_dir)
		
	# Create the actual agent file string
	agent_file = os.path.join(agent_dir, "agent.mdl")
	
	# If the agent already exists, load it
	if os.path.exists(agent_file):
		print("Agent file found.")
		return Agent(filename=agent_file)
	
	# Otherwise, create and train
	env = gym.make(environment_name)
	
	agent = Agent(network_generator)
	env_interface = EnvironmentInterface(env, render_frames=render_frames)
	
	trainer = Trainer(
		agent, env_interface,
		episodes=episodes,
		epsilon_start=epsilon_start,
		epsilon_end=epsilon_end,
		epsilon_decay=epsilon_decay,
		discount_factor=discount_factor,
		learning_rate=learning_rate,
		target_update=target_update,
		train_batch_size=train_batch_size,
		replay_memory_size=replay_memory_size,
		qvalue_approx_method=qvalue_approx_method,
		use_per=use_per,
		multi_qlearn_networks=multi_qlearn_networks,
		show_plots=show_plots,
		save_file=agent_file
	)
	
	rewards, losses, preds = trainer.train()
	train_data_file = os.path.join(agent_dir, "train_data")
	np.savez(train_data_file, rewards=rewards, losses=losses, preds=preds)
	
	# Make agent directory and save
	if not os.path.isdir(agent_dir):
		os.mkdir(agent_dir)
	
	agent.save_to_disk(agent_file)
	print("Training complete.")
	
	return agent


def render_videos(agent, env):
	"""
	Utility function for making videos to demonstrate an agent's performance during an episode.
	
	:param agent: the agent to use for action selection
	:param env: the environment to run the agent in
	"""
	print("Running agent demonstration.")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Run five episodes
	for episode in range(5):
		# Create the writer for the video file
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		video = cv2.VideoWriter(f"agent{episode}.avi", fourcc, 50, (600, 400))
		
		state = env.reset()
		while True:
			start = time.time()
			
			# Render the step to the video file
			pixels = env.render(mode="rgb_array")
			pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
			video.write(pixels)
			
			# Select action
			action = agent.dqn(
				torch.tensor([state], dtype=torch.float, device=device)
			).max(1)[1].item()
			state, reward, done, _ = env.step(action)
			
			# Quit if episode is finished
			if done:
				break
			
			end = time.time()
			time.sleep(max(0.0, 0.02 - (end - start)))
		
		# Save the episode video
		video.release()
	
	# Close the environment
	env.close()
			

if __name__ == "__main__":
	main()
