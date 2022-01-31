import math
import random
from enum import Enum
from typing import Callable

import numpy as np
import torch
from gym import Env
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import functional as F

from memory import PERMemory
from memory import ReplayMemory

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display_plots(reward_history=None, loss_history=None):
	plt.clf()
	if reward_history is not None:
		ax1 = plt.subplot(1, 2, 1)
		ax1.set_title("Training...")
		ax1.set_xlabel("Episode")
		ax1.set_ylabel("Reward")
		ax1.plot(reward_history)
		
		if len(reward_history) >= 100:
			reward_history_tensor = torch.tensor(reward_history)
			means = reward_history_tensor.unfold(0, 100, 1).mean(1).view(-1)
			means = torch.cat((torch.ones(99) * reward_history_tensor.min(0)[0], means))
			plt.plot(means.numpy())
	
	if loss_history is not None:
		ax2 = plt.subplot(1, 2, 2)
		ax2.set_title("Losses")
		ax2.set_xlabel("Episode")
		ax2.set_ylabel("Loss")
		ax2.plot(loss_history, 'r')
	
	plt.pause(0.01)


class Agent:
	def __init__(self, network_generator: Callable = None, filename: str = None):
		if filename is None:
			self.network_generator = network_generator
			self.dqn = network_generator()
		else:
			self.load_from_disk(filename)
	
	def select_action(self, state):
		return self.dqn(torch.tensor(
			[state],
			dtype=torch.float,
			device=TRAIN_DEVICE
		)).max(1)[1].view(1, 1)
	
	def save_to_disk(self, filename: str):
		torch.save(self.dqn, filename)
		
	def load_from_disk(self, filename: str):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.dqn = torch.load(filename, map_location=device)


class EnvironmentInterface:
	def __init__(self, environment: Env, render_frames: bool = True):
		self.environment = environment
		self.state = None
		self.render_frames = render_frames
	
	def reset(self):
		self.state = self.environment.reset()
	
	def step(self, action):
		state, reward, done, info = self.environment.step(action)
		self.state = state
		if self.render_frames:
			self.environment.render()
		return state, reward, done, info
	
	def get_state(self):
		return self.state
	
	def get_num_actions(self):
		return self.environment.action_space.n


class QValueApproximationMethod(Enum):
	STANDARD = 1
	DOUBLE_Q_LEARNING = 2
	MULTI_Q_LEARNING = 3


class Trainer:
	def __init__(
		self,
		agent: Agent,
		env_interface: EnvironmentInterface,
		train_batch_size: int = 128,
		discount_factor: float = 0.999,
		epsilon_start: float = 1.0,
		epsilon_end: float = 0.05,
		epsilon_decay: int = 500,
		target_update: int = 100,
		learning_rate: float = 0.001,
		episodes: int = 500,
		replay_memory_size: int = 10000,
		qvalue_approx_method: QValueApproximationMethod = QValueApproximationMethod.STANDARD,
		multi_q_learn_networks: int = 2,
		use_per: bool = False,
		clamp_grads: bool = True,
		show_plots: bool = True,
	):
		self.agent = agent
		self.env_interface = env_interface
		
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		
		self.train_episodes = episodes
		self.train_batch_size = train_batch_size
		self.discount_factor = discount_factor
		self.target_update = target_update
		self.clamp_grads = clamp_grads
		self.qvalue_approx_method = qvalue_approx_method
		self.use_per = use_per
		if use_per:
			self.replay_memory = PERMemory(replay_memory_size, episodes)
		else:
			self.replay_memory = ReplayMemory(replay_memory_size)
		
		self.optimizer = optim.Adam(self.agent.dqn.parameters(), lr=learning_rate)
		
		self.show_plots = show_plots
		
		self.state = None
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.agent.dqn.to(self.device)
		
		# If the q-value approximation method uses multi-q-learning, go ahead and create
		# the extra networks now
		if self.qvalue_approx_method == QValueApproximationMethod.MULTI_Q_LEARNING:
			self.mql_networks = []
			self.mql_optimizers = []
			self.active_network_index = 0
			for i in range(multi_q_learn_networks):
				network = self.agent.network_generator()
				self.mql_networks.append(network)
				self.mql_optimizers.append(optim.Adam(network.parameters(), lr=learning_rate))
		
		# Otherwise, just create the target dqn
		else:
			self.target = self.agent.network_generator()
			pass
		
	def select_random_action(self):
		return torch.tensor(
			[[random.randrange(self.env_interface.get_num_actions())]],
			dtype=torch.long,
			device=TRAIN_DEVICE
		)
			
	def train(self):
		# Create replay memory and other data for training
		training_step = 0
		reward_history = []  # Total reward per episode
		loss_history = []  # Average loss per episode
		
		if self.show_plots:
			plt.ion()
		
		# Start the training loop
		for episode in range(self.train_episodes):
			
			# Setup the episode
			self.env_interface.reset()
			state = self.env_interface.get_state()
			episode_reward = 0
			step_losses = []
			
			# Run the episode
			while True:
				
				# For multi q-learning, we need to swap the "active" network
				# for each step
				if self.qvalue_approx_method == QValueApproximationMethod.MULTI_Q_LEARNING:
					self.active_network_index = random.randrange(len(self.mql_networks))
					self.agent.dqn = self.mql_networks[self.active_network_index]
					self.optimizer = self.mql_optimizers[self.active_network_index]
				
				# Take an action based on the current network state
				sample = random.random()
				epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) \
							* math.exp(-1. * training_step / self.epsilon_decay)
				
				# Use epsilon-greedy policy
				if sample > epsilon:
					with torch.no_grad():
						action = self.agent.select_action(state)
				else:
					action = self.select_random_action()
				
				# Step the simulation
				_, reward, done, _ = self.env_interface.step(action.item())
				episode_reward += reward
				
				# Get the next state
				next_state = self.env_interface.get_state()
				
				# Store transition
				self.replay_memory.push((
					torch.tensor([state], dtype=torch.float, device=self.device),
					action,
					torch.tensor([next_state], dtype=torch.float, device=self.device),
					torch.tensor([reward], dtype=torch.float, device=self.device),
					done
				))
				
				# Optimize the model
				loss = self.optimize_model()
				step_losses.append(loss)
				
				# Update the training step
				state = next_state
				training_step += 1
				
				# Check if the target network should be updated
				if self.qvalue_approx_method != QValueApproximationMethod.MULTI_Q_LEARNING and \
						training_step % self.target_update == 0:
					print("Updating target network.")
					self.update_target_network()
				
				# Check episode finished
				if done:
					break
		
			# Update PER beta if necessary
			# if self.use_per:
			# 	self.replay_memory.step_episode()
		
			# Print training information
			print(f"Episode {episode} "
				f"(e = {epsilon:0.2f}) "
				f"{f'(b = {self.replay_memory.beta:0.2f}) ' if self.use_per else ''}"
				f"[Mem: {float(len(self.replay_memory)) / self.replay_memory.max_size() * 100:0.2f}%] ")
			
			reward_history.append(episode_reward)
			loss_history.append(np.average(np.array(step_losses)))
			
			if self.show_plots:
				loss_history_np = np.array(loss_history)
				max_val = loss_history_np.max()
				loss_history_np = np.where(loss_history_np < 0, max_val, loss_history_np)
				display_plots(reward_history, loss_history_np)
			
		if self.show_plots:
			plt.ioff()
			plt.show()  # Use show here to block until the user closes it
		
	def update_target_network(self):
		self.target.load_state_dict(self.agent.dqn.state_dict())
	
	def optimize_model(self):
		# Check that there are enough entries in replay memory to train
		if len(self.replay_memory) < self.train_batch_size:
			return -1
		
		# Load a random sample of transitions from memory
		if self.use_per:
			transitions, indexes, is_weights = self.replay_memory.sample(self.train_batch_size)
			is_weights = torch.tensor(
				is_weights,
				dtype=torch.int64,
				device=TRAIN_DEVICE
			)
		else:
			transitions = self.replay_memory.sample(self.train_batch_size)
			
		state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*transitions)
		
		# Divide into individual batches
		state_batch = torch.cat(state_batch)
		action_batch = torch.cat(action_batch)
		reward_batch = torch.cat(reward_batch)
		next_state_batch = torch.cat(next_state_batch)
		done_batch = [done for done in done_batch]
		non_terminal_states = [not done for done in done_batch]
		
		# Compute the q-values: Q(s, a)
		q_values = self.agent.dqn(state_batch).gather(1, action_batch)
		
		# Find the values of the next states
		next_state_values = torch.zeros(self.train_batch_size, device=self.device)
		
		# Standard Q value approximation
		if self.qvalue_approx_method == QValueApproximationMethod.STANDARD:
			next_state_values[non_terminal_states] = \
				self.target(next_state_batch).detach().max(1)[0][non_terminal_states]
		
		# Double Q Learning State value approximation
		# This code uses the double dqn approach to compute expected state values
		# Find the actions our network would take in the new state
		elif self.qvalue_approx_method == QValueApproximationMethod.DOUBLE_Q_LEARNING:
			next_state_actions = self.agent.dqn(next_state_batch).max(1)[1].unsqueeze(1)
			next_state_values[non_terminal_states] = \
				self.target(next_state_batch).detach().gather(1, next_state_actions)[non_terminal_states].squeeze(1)
		
		# Multi Q Learning State value approximation
		elif self.qvalue_approx_method == QValueApproximationMethod.MULTI_Q_LEARNING:
			next_state_actions = self.agent.dqn(next_state_batch).max(1)[1].unsqueeze(1)
			
			# Sum the values provided by all other networks.
			for i in range(len(self.mql_networks)):
				if i != self.active_network_index:
					self.mql_networks[i].eval()
					
					with torch.no_grad():
						next_state_values[non_terminal_states] = next_state_values[non_terminal_states].add(
							self.mql_networks[i](next_state_batch).detach().gather(1, next_state_actions)[
								non_terminal_states].squeeze(1)
						)
					
					self.mql_networks[i].train()
			
			# Take the average
			next_state_values = next_state_values.div(len(self.mql_networks) - 1)
		
		# Invalid value
		else:
			pass
		
		# The expected q-value: E[r + discount_factor * max[a]Q(s', a)]
		expected_q_values = reward_batch + (self.discount_factor * next_state_values)
		
		# For PER, update the memory priorities
		if self.use_per:
			errors = torch.abs(q_values - expected_q_values.unsqueeze(1)).detach().cpu().numpy()
			for i in range(self.train_batch_size):
				index = indexes[i]
				self.replay_memory.update(index, errors[i])
		
		# Find loss using Huber Loss (smooth l1 loss is Huber loss with delta = 1)
		loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
		
		# For PER, scale based on the importance-sampling weights
		if self.use_per:
			loss = (is_weights * loss).mean()
		
		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		if self.clamp_grads:
			for param in self.agent.dqn.parameters():
				param.grad.data.clamp_(-1, 1)  # Clamp gradient to prevent the exploding gradient problem
		
		self.optimizer.step()
		
		return loss.item()
	