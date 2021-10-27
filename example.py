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
import time

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
EPS_DECAY = 200
TARGET_UPDATE = 500
NUM_EPISODES = 200
LEARNING_RATE = 1e-2


def main():
	# Get the model name
	if len(sys.argv) < 2:
		print("Please provide model name.")
		sys.exit(0)

	model_name = sys.argv[1]
	model_file = f"{model_name}.mdl"

	# Create the gym environment
	env = gym.make("CartPole-v0")

	# Load model if exists
	if exists(model_file):
		print("Model found. Running model.")
		model = torch.load(model_file, map_location=torch.device(device))
		run_model(model, env)
		sys.exit(0)
	else:
		print(f"No model found. Training new model: {model_name}")

	# Setup interactive mode for matplotlib
	plt.ion()

	# Setup input/output spaces
	input_shape = (40, 60, 1)  # Scaled down 400x600 RGB image
	num_inputs = env.observation_space.shape[0]
	num_actions = env.action_space.n

	# Build the DQN (separate target and policy nets)
	policy_network = create_dqn(num_inputs, num_actions)
	target_network = create_dqn(num_inputs, num_actions)
	target_network.load_state_dict(policy_network.state_dict())
	target_network.eval()
	print("Using network structure:")
	print(policy_network, "\n")

	optimizer = optim.RMSprop(policy_network.parameters(), lr=LEARNING_RATE)

	# Setup the replay memory
	memory = ReplayMemory(REPLAY_MEMORY_SIZE)
	print(f"Replay memory initialized with max size {REPLAY_MEMORY_SIZE}")

	# Generate some random play to have initial memory
	step = 0
	while step < 128:
		state = env.reset()
		for _ in count():
			action = torch.tensor(random.randrange(num_actions), device=device, dtype=torch.long)
			next_state, reward, done, _ = env.step(action.item())
			reward = torch.tensor([reward], device=device)
			
			memory.push(
				torch.tensor([state], device=device),
				torch.tensor([action], device=device),
				torch.tensor([next_state], device=device) if not done else None,
				reward
			)
			step += 1
			print(f"Random Memory {step}/1000")
			if done or step >= 128:
				break;

	# Startup the actual training process
	step = 0
	reward_tracker = []

	for i_episode in range(NUM_EPISODES):

		# Reset for this episode
		state = env.reset()
		episode_reward = 0

		for _ in count():
			# Take an action based on the current state
			action = select_action(policy_network, state, step, num_actions)

			next_state, reward, done, _ = env.step(action.item())
			episode_reward += reward
			reward = torch.tensor([reward], device=device)

			env.render()

			# Store the state transition in memory
			memory.push(
				torch.tensor([state], device=device),
				torch.tensor([action], device=device),
				torch.tensor([next_state], device=device) if not done else None,
				reward
			)

			# Move to next state
			state = next_state

			# Run one step of the optimizer
			optimize_model(memory, policy_network, target_network, optimizer)

			# Add step
			step += 1
			
			if step % TARGET_UPDATE == 0:
				target_network.load_state_dict(policy_network.state_dict())
				print("Updating target network.")

			if done:
				break

		mem_full = len(memory) / REPLAY_MEMORY_SIZE
		mem_empt = 1 - mem_full
		loading = "[" + ("=" * int(math.floor(50 * mem_full))) + ("_" * int(math.ceil(50 * mem_empt))) + "]"
		print(f"Memory: {loading} {len(memory)} ({mem_full * 100:.2f}%)")
		sys.stdout.flush()

		reward_tracker.append(episode_reward)

		# Plot the reward for this ep
		plt.figure(2)
		plt.clf()
		plt.title("Training...")
		plt.xlabel("Episode")
		plt.ylabel("Reward")
		plt.plot(reward_tracker)
		plt.pause(0.01)

		# Take 100 episode averages and plot them
		if len(reward_tracker) >= 100:
			reward_tracker_tensor = torch.tensor(reward_tracker)
			means = reward_tracker_tensor.unfold(0, 100, 1).mean(1).view(-1)
			means = torch.cat((torch.ones(99) * reward_tracker_tensor.min(0)[0], means))
			plt.plot(means.numpy())

	print("Training complete. Saving model.")
	torch.save(policy_network, model_file)
	run_model(policy_network, env)


def get_screen(rgb_array):
	rgb_array = rgb_array.copy()
	resized = cv2.resize(rgb_array, (60, 40))
	resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
	state = torch.tensor([resized], device=device, dtype=torch.float) / 255.0
	state = state.unsqueeze(0)
	return state, resized


def conv_output_size(input_shape, kernel_size, padding=0, stride=1):
	return np.floor((input_shape + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


def create_dqn(input_shape, num_outputs):
	# channels = input_shape[2]
	# shape = (input_shape[0], input_shape[1])
	# conv1_output_shape = conv_output_size(np.array(shape), 5, stride=2)
	# conv2_output_shape = conv_output_size(conv1_output_shape, 5, stride=2)
	# conv3_output_shape = conv_output_size(conv2_output_shape, 5, stride=2)
	# linear_input_size = int(conv3_output_shape[0] * conv3_output_shape[1]) * 32

	dqn = nn.Sequential(
		nn.Linear(input_shape, 128, bias=0.01),
		nn.ReLU(),
		nn.Linear(128, 64, bias=0.01),
		nn.ReLU(),
		nn.Linear(64, 32, bias=0.01),
		nn.ReLU(),
		nn.Linear(32, num_outputs)
	).to(device)
	
	def init_weights(m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_normal(m.weight)
	
	dqn.apply(init_weights)

	return dqn


def select_action(policy, state, step, actions):
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * step / EPS_DECAY)
	if sample > eps_threshold:
		with torch.no_grad():
			state = torch.tensor(state, device=device)
			return policy(state).max(0)[1].view(1, 1)
	else:
		return torch.tensor(random.randrange(actions), device=device, dtype=torch.long)


def optimize_model(memory, policy_network, target_network, optimizer):
	if len(memory) < BATCH_SIZE:
		return

	# Load batch transitions from memory
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	# Determine which transitions were to a final state
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

	# Divide states, actions, rewards into individual batches
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Computes Q(s_t, a)
	# Find the value of the action our policy would take in these states
	state_action_values = policy_network(state_batch).gather(1, action_batch.unsqueeze(1).long())

	# Computes V(s_{t+1}) for all next states
	# This finds the maximum expected value across all possible
	# actions in the next states.
	# This is merged based on the mask, so that we have the expected
	# state value or 0 if it was a final state.
	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()

	# Computes the expected Q-values
	# E[r + y(max_a(Q(s', a)))]
	expected_state_action_values = reward_batch + (next_state_values * GAMMA)

	# Find the loss using Huber loss function
	loss_fn = nn.HuberLoss()
	loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

	# Optimize the model based on loss
	optimizer.zero_grad()
	loss.backward()
	for param in policy_network.parameters():
		param.grad.data.clamp_(-1, 1)  # Clamp the gradient to prevent exploding grad problem

	optimizer.step()


def run_model(model, env):
	for i_episode in range(20):
		# Start over
		state = env.reset()

		for t in count():
			# Select the action and update the environment
			action = model(torch.tensor(state, device=device)).max(0)[1].view(1, 1)
			state, reward, done, _ = env.step(action.item())

			# Render the new state
			env.render()

			# Break if the episode is finished
			if done:
				break

	env.close()


if __name__ == "__main__":
	main()
