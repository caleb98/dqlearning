import os
import time

import cv2
import gym
import torch.cuda

import networks
from training import Agent
from training import VisualTrainer


def main():
	environment = "LunarLander-v2"
	agent_file = f"{environment}-vis-agent.mdl"
	env = gym.make(environment)
	
	agent = None
	
	# Try to load agent from disk
	if os.path.exists(agent_file):
		print("Agent file found.")
		agent = Agent(filename=agent_file)
	
	# Create and train agent
	else:
		print("No agent found. Creating new agent and training.")
		width = 75
		height = 50
		num_actions = env.action_space.n
		
		network = networks.ConvolutionalNetwork(1, width, height, num_actions)
		network_target = networks.ConvolutionalNetwork(1, width, height, num_actions)
		network_target.eval()
		
		print("Network:")
		print(network)
		
		agent = Agent(network, network_target)
		
		trainer = VisualTrainer(
			agent,
			env,
			render_size=(width, height),
			episodes=1000,
			epsilon_start=1.0,
			epsilon_end=0.05,
			epsilon_decay=10000,
			discount_factor=0.99,
			learning_rate=0.01,
			target_update=250,
			train_batch_size=128,
			replay_memory_size=1000000
		)
		trainer.train()
		agent.save_to_disk(agent_file)
		print("Training complete.")
	
	input("Press any key to run agent...")
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
			time.sleep(max(0, 0.02 - (end - start)))
		
		video.release()
	
	env.close()


if __name__ == "__main__":
	main()
