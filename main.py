import os
import time

import cv2
import gym
import torch.cuda

import networks
from training import Agent
from training import EnvironmentInterface
from training import QValueApproximationMethod
from training import Trainer


def main():
	environment = "LunarLander-v2"
	agent_file = f"{environment}-agent.mdl"
	env = gym.make(environment)
	
	# Try to load agent from disk
	if os.path.exists(agent_file):
		print("Agent file found.")
		agent = Agent(filename=agent_file)
		
	# Create and train agent
	else:
		print("No agent found. Creating new agent and training.")
		num_inputs = 1
		for i in env.observation_space.shape:
			num_inputs *= i
		num_actions = env.action_space.n
	
		def network_generator():
			network = networks.LinearNetwork(num_inputs, num_actions)
			network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
			return network
	
		agent = Agent(network_generator)
		env_interface = EnvironmentInterface(env, render_frames=True)
		
		trainer = Trainer(
			agent,
			env_interface,
			episodes=250,
			epsilon_start=1.0,  # 1.0 is good
			epsilon_end=0.01,  # 0.01
			epsilon_decay=10000,  # ~10,000 good for standard/double dqn
			discount_factor=0.99,  # Tune depending on task
			learning_rate=0.001,  # ~0.0001 good for standard/double dqn. Increase for PER
			target_update=500,
			train_batch_size=128,
			replay_memory_size=250000,
			qvalue_approx_method=QValueApproximationMethod.DOUBLE_Q_LEARNING,
			use_per=True,
			multi_q_learn_networks=8
		)
		trainer.train()
		agent.save_to_disk(agent_file)
		print("Training complete.")
	
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
