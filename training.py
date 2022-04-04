"""
Contains classes and functions used for training agents in OpenAI gym environments.
"""

import math
import random
import string
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

# Set the training device to the GPU if possible.
TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display_plots(reward_history=None, loss_history=None, preds_history=None):
	"""Utility function for displaying and updating matplotlib plots during agent training."""
	
	# Clear the plot
	plt.clf()
	
	# Add the reward history plot
	if reward_history is not None:
		ax1 = plt.subplot(1, 3, 1)
		ax1.set_title("Training...")
		ax1.set_xlabel("Episode")
		ax1.set_ylabel("Reward")
		ax1.plot(reward_history)
		
		# If we've ran for at least 100 episodes, also plot the past 100 episode mean reward
		if len(reward_history) >= 100:
			reward_history_tensor = torch.tensor(reward_history)
			means = reward_history_tensor.unfold(0, 100, 1).mean(1).view(-1)
			means = torch.cat((torch.ones(99) * reward_history_tensor.min(0)[0], means))
			plt.plot(means.numpy())
	
	# Add the loss history plot
	if loss_history is not None:
		ax2 = plt.subplot(1, 3, 2)
		ax2.set_title("Losses")
		ax2.set_xlabel("Episode")
		ax2.set_ylabel("Loss")
		ax2.plot(loss_history, 'r')

	# Add the prediction history plot
	if preds_history is not None:
		ax3 = plt.subplot(1, 3, 3)
		ax3.set_title("Q Predictions")
		ax3.set_xlabel("Episode")
		ax3.set_ylabel("Predicted Reward")
		ax3.plot(preds_history)
	
	plt.pause(0.01)


class Agent:
	"""
	An agent that may take actions in an environment. Contains the underlying model and functionality for action
	selection.
	"""
	
	def __init__(self, network_generator: Callable = None, filename: str = None):
		if filename is None:
			self.network_generator = network_generator
			self.dqn = network_generator()
		else:
			self.load_from_disk(filename)
	
	def select_action(self, state):
		"""Uses this agent's model to select an action given the current state."""
		return self.dqn(torch.tensor(
			[state],
			dtype=torch.float,
			device=TRAIN_DEVICE
		)).max(1)[1].view(1, 1)
	
	def save_to_disk(self, filename: str):
		"""Save's this agent's underlying model to the given file."""
		torch.save(self.dqn, filename)
		
	def load_from_disk(self, filename: str):
		"""Loads into this agent the model in the specified file."""
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.dqn = torch.load(filename, map_location=device)


class EnvironmentInterface:
	"""
	Wrapper class for for an OpenAI gym environment used to facilitate better modularity within the agent Trainer
	class.
	"""
	
	def __init__(self, environment: Env, render_frames: bool = True):
		self.environment = environment
		self.state = None
		self.render_frames = render_frames
	
	def reset(self):
		"""Resets the environment for a new episode."""
		self.state = self.environment.reset()
	
	def step(self, action):
		"""
		Takes a step in the environment using the given action
		
		:param action: the action to take
		:return: state, reward, done, info:
			The new state;
			The reward obtained for taking the action;
			Whether the episode is finished;
			Debug info
		"""
		state, reward, done, info = self.environment.step(action)
		self.state = state
		if self.render_frames:
			self.environment.render()
		return state, reward, done, info
	
	def get_state(self):
		"""Returns the current state of the environment."""
		return self.state
	
	def get_num_actions(self):
		"""Returns the number of actions that may be takne in the environment."""
		return self.environment.action_space.n


class QValueApproximationMethod(Enum):
	"""Enumeration of valid q-value approximation methods."""
	STANDARD = "Standard"
	DOUBLE_Q_LEARNING = "DoubleDQN"
	MULTI_Q_LEARNING = "MultiQ"
	
	def __str__(self):
		return self.value


class Trainer:
	"""Manages an environmental agent and handles the training of the agent's backing network."""
	
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
		multi_qlearn_networks: int = 2,
		use_per: bool = False,
		clamp_grads: bool = True,
		show_plots: bool = True,
		save_file: string = None,
		save_eps: int = 10
	):
		"""
		:param agent: The agent containing the model to be trained
		:param env_interface: An interface to the training environment
		:param train_batch_size: Number of transitions to sample from replay memory in each training step
		:param discount_factor: How strongly future rewards are considered
		:param epsilon_start: Starting value for the epsilon used in epsilon-greedy action selection
		:param epsilon_end: Ending value for the epsilon used in epsilon-greedy action selection
		:param epsilon_decay: Over how many training steps to decay epsilon from epsilon_start to epsilon_end
		:param target_update: Number of training steps to update the target network after (for double dqn)
		:param learning_rate: How strongly to update the network based on training loss
		:param episodes: Number of episodes to run the training for
		:param replay_memory_size: Maximum number of transitions to store in replay memory
		:param qvalue_approx_method: The method that should be used to approximate q-values using the agent's network
		:param multi_qlearn_networks: Number of networks to use if the qvalue_approx_method is MULTI_Q_LEARNING
		:param use_per: Whether or not PER should be used when sampling from replay memory
		:param clamp_grads: Whether or not to clamp network gradients during training (can prevent exploding gradient problem)
		:param show_plots: Whether or not to display plots live during the training process.
		:param save_file: The file to save the model to (used to make periodic backups during the training process.)
		:param save_eps: Episode interval used to save model backups
		"""
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
		
		self.save_file = save_file
		self.save_eps = save_eps
		
		self.state = None
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.agent.dqn.to(self.device)
		
		# If the q-value approximation method uses multi-q-learning, go ahead and create
		# the extra networks now
		if self.qvalue_approx_method == QValueApproximationMethod.MULTI_Q_LEARNING:
			self.mql_networks = []
			self.mql_optimizers = []
			self.active_network_index = 0
			for i in range(multi_qlearn_networks):
				network = self.agent.network_generator()
				self.mql_networks.append(network)
				self.mql_optimizers.append(optim.Adam(network.parameters(), lr=learning_rate))
		
		# Otherwise, just create the target dqn
		else:
			self.target = self.agent.network_generator()
			pass
	
	def select_random_action(self):
		"""Selects a random action for the provided environment."""
		return torch.tensor(
			[[random.randrange(self.env_interface.get_num_actions())]],
			dtype=torch.long,
			device=TRAIN_DEVICE
		)
			
	def train(self):
		"""
		Trains the agent supplied to this trainer based on the specified hyperparameters.
		
		:returns reward_history (ndarray), loss_history (ndarray), prediction_history (ndarray):
			An array containing the historical reward values during training;
			An array containing the historical network losses during training;
			An array containing the historical q-value predictions during training
		"""
		
		# Create replay memory and other data for training
		training_step = 0
		reward_history = []  # Total reward per episode
		loss_history = []  # Average loss per tra
		predicted_reward_history = []  # Average predicted reward
		
		# Turn on interactive mode for plots if plots are displayed
		if self.show_plots:
			plt.ion()
		
		# Start the training loop
		for episode in range(self.train_episodes):
			
			# Setup the episode
			self.env_interface.reset()
			state = self.env_interface.get_state()
			episode_reward = 0
			step_losses = []
			step_preds = []
			
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
				loss, preds = self.optimize_model()
				step_losses.append(loss)
				step_preds.append(preds)
				
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
			if self.use_per:
				self.replay_memory.step_episode()
		
			# Print training information
			print(f"Episode {episode} "
				f"(e = {epsilon:0.2f}) "
				f"{f'(b = {self.replay_memory.beta:0.2f}) ' if self.use_per else ''}"
				f"[Mem: {float(len(self.replay_memory)) / self.replay_memory.max_size() * 100:0.2f}%] ")
			
			# Update the historical data
			reward_history.append(episode_reward)
			loss_history.append(np.average(np.array(step_losses)))
			predicted_reward_history.append(np.average(np.array([val for val in step_preds if val is not None])))
			
			# Save the network on the requested steps
			if episode != 0 and episode % self.save_eps == 0:
				print(f"Saving model backup for episode {episode}.")
				self.agent.save_to_disk(f"{self.save_file}_episode{episode}.mdl")
			
			# Update the plots if
			if self.show_plots:
				loss_history_np = np.array(loss_history)
				max_val = loss_history_np.max()
				loss_history_np = np.where(loss_history_np < 0, max_val, loss_history_np)

				preds_history_np = np.array(predicted_reward_history)
				min_val = preds_history_np.min()
				preds_history_np = np.where(preds_history_np is None, min_val, preds_history_np)
				display_plots(reward_history, loss_history_np, preds_history_np)
		
		# Display the plots if requested
		if self.show_plots:
			plt.ioff()
			plt.show()  # Use show here to block until the user closes it
		
		# Calculate the loss and prediction history
		loss_history_np = np.array(loss_history)
		max_val = loss_history_np.max()
		loss_history_np = np.where(loss_history_np < 0, max_val, loss_history_np)
		
		preds_history_np = np.array(predicted_reward_history)
		min_val = preds_history_np.min()
		preds_history_np = np.where(preds_history_np is None, min_val, preds_history_np)
		
		# Return the historical values throughout training (so they can be saved, graphed, etc.)
		return np.array(reward_history), loss_history_np, preds_history_np
		
	def update_target_network(self):
		"""
		Updates the target model using the weights from the active model.
		Only works when the q-value approximation method is set to DOUBLE_Q_LEARNING.
		"""
		self.target.load_state_dict(self.agent.dqn.state_dict())
	
	def optimize_model(self):
		"""
		Performs a single training step in the deep q-learning algorithm.
		
		This training step will only be performed if a sufficient number of training samples are present in the replay
		memory to satisfy the training batch size.
		Checks are made so that the training step is carried out using the appropriate q-value approximation method and
		using PER for transition sampling if enabled.
		"""
		# Check that there are enough entries in replay memory to train
		if len(self.replay_memory) < self.train_batch_size:
			return -1, None
		
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
		
		return loss.item(), expected_q_values.mean().item()