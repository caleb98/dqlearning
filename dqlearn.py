import gym
import math

import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import cv2

from os.path import exists
from itertools import count
from replay_memory import ReplayMemory, Transition


# Setup cuda device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")
sys.stdout.flush()

# Training hyperparameters
REPLAY_MEMORY_SIZE = 25000
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 10
NUM_EPISODES = 150
LEARNING_RATE = 1e-2


def main():
	# Get the model name
	if len(sys.argv) < 2:
		print("Please provide model name.")
		sys.exit(0)

	model_name = sys.argv[1]
	model_file = f"{model_name}.mdl"

	# Create the gym environment
	env = gym.make("LunarLander-v2")

	# Load model if exists
	if exists(model_file):
		print("Model found. Running model.")
		model = torch.load(model_file, map_location=torch.device(device))
		run_model(model, env)
		sys.exit(0)
	else:
		print(f"No model found. Training new model: {model_name}")

	plt.ion()

	# Setup input/output spaces
	input_shape = (40, 60, 3)  # Scaled down 400x600 RGB image
	num_actions = env.action_space.n

	# Build the DQN (separate target and policy nets)
	policy_network = create_dqn(input_shape, num_actions)
	target_network = create_dqn(input_shape, num_actions)
	target_network.load_state_dict(policy_network.state_dict())
	target_network.eval()
	print("Using network structure:")
	print(policy_network, "\n")

	optimizer = optim.RMSprop(policy_network.parameters(), lr=LEARNING_RATE)

	# Setup the replay memory
	memory = ReplayMemory(REPLAY_MEMORY_SIZE)
	print(f"Replay memory initialized with max size {REPLAY_MEMORY_SIZE}")

	step = 0
	reward_tracker = []

	for i_episode in range(NUM_EPISODES):

		# Reset for this episode
		env.reset()
		state = env.render(mode="rgb_array")
		state = cv2.resize(state, (60, 40))
		state = torch.tensor(state, device=device)
		episode_reward = 0

		for t in count():
			# Take an action based on the current state
			action = select_action(policy_network, state, step, num_actions)
			observation, reward, done, _ = env.step(action.item())
			episode_reward += reward
			reward = torch.tensor([reward], device=device)

			next_state = env.render(mode="rgb_array")
			next_state = cv2.resize(next_state, (60, 40)).transpose(1, 2).transpose(0, 1)


			# Store the state transition in memory
			memory.push(
				torch.tensor([state], device=device),
				torch.tensor([[action]], device=device),
				torch.tensor([next_state], device=device) if not done else None,
				torch.tensor([reward], device=device)
			)

			# Move to next state
			state = next_state

			# Run one step of the optimizer
			optimize_model(memory, policy_network, target_network, optimizer)

			if done:
				break

		mem_full = len(memory) / REPLAY_MEMORY_SIZE
		mem_empt = 1 - mem_full
		loading = "[" + ("=" * int(math.floor(50 * mem_full))) + ("_" * int(math.ceil(50 * mem_empt))) + "]"
		print(f"Memory: {loading} {len(memory)} ({mem_full * 100:.2f}%)")
		sys.stdout.flush()

		if i_episode % TARGET_UPDATE == 0:
			target_network.load_state_dict(policy_network.state_dict())

		reward_tracker.append(episode_reward)

		# Plot the reward for this ep
		plt.figure(2)
		plt.clf()
		plt.title("Training...")
		plt.xlabel("Episode")
		plt.ylabel("Reward")
		plt.plot(reward_tracker)
		plt.pause(0.001)

		# Take 100 episode averages and plot them
		if len(reward_tracker) >= 100:
			reward_tracker_tensor = torch.tensor(reward_tracker)
			means = reward_tracker_tensor.unfold(0, 100, 1).mean(1).view(-1)
			means = torch.cat((torch.ones(99) * reward_tracker_tensor.min(0)[0], means))
			plt.plot(means.numpy())

	print("Training complete. Saving model.")
	torch.save(target_network, model_file)
	run_model(target_network, env)


def conv_output_size(input_shape, kernel_size, padding=0, stride=1):
	return np.floor((input_shape + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


def create_dqn(input_shape, num_outputs):
	channels = input_shape[2]
	shape = (input_shape[0], input_shape[1])
	conv1_output_shape = conv_output_size(np.array(shape), 5)
	conv2_output_shape = conv_output_size(conv1_output_shape, 5)
	linear_input_size = int(conv2_output_shape[0] * conv2_output_shape[1]) * 16

	return nn.Sequential(
		nn.Conv2d(channels, 8, kernel_size=5),
		nn.Conv2d(8, 16, kernel_size=5),
		nn.Flatten(),
		nn.Linear(linear_input_size, 128),
		nn.ReLU(),
		nn.Linear(128, 64),
		nn.ReLU(),
		nn.Linear(64, 32),
		nn.ReLU(),
		nn.Linear(32, num_outputs)
	).to(device)


def select_action(policy, state, step, actions):
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * step / EPS_DECAY)

	if sample > eps_threshold:
		state = state.copy()
		state_tensor = torch.tensor(state, device=device, dtype=torch.float)
		state_tensor = state_tensor.unsqueeze(0)
		with torch.no_grad():
			result = policy(state_tensor)
			return result.max(1)[1]
	else:
		return torch.tensor(random.randrange(actions), device=device, dtype=torch.int16)


def optimize_model(memory, policy_network, target_network, optimizer):
	if len(memory) < BATCH_SIZE:
		return

	# Load batch transitions from memory
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	# Determine which transitions were to a final state
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

	# Divide states, actions, rewards into individual batches
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Fix state batch structure
	state_batch = state_batch.transpose(2, 3).transpose(1, 2)

	# Computes Q(s_t, a)
	# Find what action our policy would take in these states
	state_action_values = policy_network(state_batch).gather(1, action_batch)

	# Computes V(s_{t+1}) for all next states
	# This finds the maximum expected value across all possible
	# actions in the next states.
	# This is merged based on the mask, so that we have the expected
	# state value or 0 if it was a final state.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()

	# Computes the expected Q-values
	# E[r + y(max_a(Q(s', a)))]
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Find the loss using Huber loss function
	loss_fn = nn.SmoothL1Loss()
	loss = loss_fn(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

	# Optimize the model based on loss
	optimizer.zero_grad()
	loss.backward()
	for param in policy_network.parameters():
		param.grad.data.clamp_(-1, 1)  # Clamp the gradient to prevent exploding grad problem

	optimizer.step()


def run_model(model, env):
	for i_episode in range(50):
		# Start over
		state = env.reset()

		for t in count():
			state = torch.tensor(state, device=device)
			with torch.no_grad():
				action = model(state).max(0)[1]
				state, reward, done, _ = env.step(action.item())
				env.render()

				if done:
					break


if __name__ == "__main__":
	main()
