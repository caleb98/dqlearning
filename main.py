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

AGENTS_DIR = "./agents"


def main():
	# Create the agents directory if necessary
	if not os.path.isdir(AGENTS_DIR):
		os.mkdir(AGENTS_DIR)
	
	environment = "LunarLander-v2"
	env = gym.make(environment)

	num_inputs = 1
	for i in env.observation_space.shape:
		num_inputs *= i
	num_actions = env.action_space.n

	def network_generator():
		network = networks.LinearNetwork(num_inputs, num_actions)
		network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		return network
	
	train_agent(
		environment,
		network_generator,
		qvalue_approx_method=QValueApproximationMethod.STANDARD,
		use_per=False,
		multi_qlearn_networks=8,
		render_frames=False,
		show_plots=False,
		episodes=10
	)


def train_agent(environment_name, network_generator, qvalue_approx_method: QValueApproximationMethod, use_per: bool,
				multi_qlearn_networks: int = 8,	render_frames: bool = True, episodes: int = 250,
				epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: int = 10000,
				discount_factor: float = 0.99, learning_rate: float = 0.001, target_update: int = 500,
				train_batch_size: int = 128, replay_memory_size: int = 250000, show_plots: bool = True):
	
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
		show_plots=show_plots
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
	print("Running agent demonstration.")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	for episode in range(5):
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		video = cv2.VideoWriter(f"agent{episode}.avi", fourcc, 50, (600, 400))
		
		state = env.reset()
		while True:
			start = time.time()
			
			# Render
			pixels = env.render(mode="rgb_array")
			pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
			video.write(pixels)
			
			# Select action
			action = agent.dqn(
				torch.tensor([state], dtype=torch.float, device=device)
			).max(1)[1].item()
			state, reward, done, _ = env.step(action)
			
			if done:
				break
			
			end = time.time()
			time.sleep(max(0.0, 0.02 - (end - start)))
		
		video.release()
	
	env.close()
			

if __name__ == "__main__":
	main()
