import math
import random
from abc import ABC
from abc import abstractmethod
from typing import Tuple

import cv2
import numpy as np
import torch
from gym import Env
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.nn import functional as F

from util import ReplayMemory


class Agent:
	def __init__(self, network: nn.Module = None, target: nn.Module = None, filename: str = None):
		if filename is None:
			self.dqn = network
			self.dqn_target = target
			self.update_target_network()
		else:
			self.load_from_disk(filename)
	
	def select_action(self, state):
		return self.dqn(state).max(1)[1]

	def update_target_network(self):
		self.dqn_target.load_state_dict(self.dqn.state_dict())
	
	def save_to_disk(self, filename: str):
		torch.save(self.dqn, filename)
		
	def load_from_disk(self, filename: str):
		device = torch.device("cuda" if torch.cuda.is_available() else  "cpu")
		self.dqn = torch.load(filename, map_location=device)
		self.dqn_target = torch.load(filename, map_location=device)
		self.dqn_target.eval()
		self.update_target_network()
	
	
class Trainer(ABC):
	def __init__(
		self,
		agent: Agent,
		environment: Env,
		train_batch_size: int = 128,
		discount_factor: float = 0.999,
		epsilon_start: float = 1.0,
		epsilon_end: float = 0.05,
		epsilon_decay: int = 500,
		target_update: int = 100,
		learning_rate: float = 0.01,
		episodes: int = 250,
		replay_memory_size: int = 10000,
		show_plots: bool = True
	):
		self.agent = agent
		self.environment = environment
		
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		
		self.replay_memory = ReplayMemory(replay_memory_size)
		self.train_episodes = episodes
		self.train_batch_size = train_batch_size
		self.discount_factor = discount_factor
		self.target_update = target_update
		
		self.optimizer = optim.Adam(self.agent.dqn.parameters(), lr=learning_rate)
		
		self.show_plots = show_plots
		
		self.state = None
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.agent.dqn.to(self.device)
		self.agent.dqn_target.to(self.device)
	
	@abstractmethod
	def reset_environment(self):
		pass
	
	@abstractmethod
	def step(self, action):
		pass

	@abstractmethod
	def get_state(self):
		pass
	
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
			self.reset_environment()
			state = self.get_state()
			episode_reward = 0
			step_losses = []
			
			# Run the episode
			while True:
				
				# Take an action based on the current network state
				sample = random.random()
				epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) \
							* math.exp(-1. * training_step / self.epsilon_decay)
				
				# Use epsilon-greedy policy
				if sample > epsilon:
					with torch.no_grad():
						action = self.agent.dqn(torch.tensor(
							[state],
							dtype=torch.float,
							device=self.device
						)).max(1)[1].view(1, 1)
				else:
					action = torch.tensor(
						[[random.randrange(self.environment.action_space.n)]],
						dtype=torch.long,
						device=self.device
					)
				
				# Step the simulation
				_, reward, done, _ = self.step(action.item())
				episode_reward += reward
				
				# Get the next state
				next_state = self.get_state()
				
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
				if training_step % self.target_update == 0:
					self.agent.update_target_network()
				
				# Check episode finished
				if done:
					break
		
			print(f"Episode {episode} "
				f"(e = {epsilon:0.2f}) "
				f"[Mem: {float(len(self.replay_memory)) / self.replay_memory.max_size() * 100:0.2f}%] ")
			
			reward_history.append(episode_reward)
			loss_history.append(np.average(np.array(step_losses)))
			
			if self.show_plots:
				plt.clf()
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
				
				loss_history_np = np.array(loss_history)
				max_val = loss_history_np.max()
				if training_step > self.train_batch_size:
					print(end="")
				loss_history_np = np.where(loss_history_np < 0, max_val, loss_history_np)
				ax2 = plt.subplot(1, 2, 2)
				ax2.set_title("Losses")
				ax2.set_xlabel("Episode")
				ax2.set_ylabel("Loss")
				ax2.plot(loss_history_np, 'r')
				
				plt.pause(0.01)
			
		if self.show_plots:
			plt.ioff()
		
		self.agent.update_target_network()
		
	def optimize_model(self):
		# Check that there are enough entries in replay memory to train
		if len(self.replay_memory) < self.train_batch_size:
			return -1
		
		# Load a random sample of transitions from memory
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
		next_state_values = torch.zeros(self.train_batch_size)
		next_state_values[non_terminal_states] = self.agent.dqn_target(next_state_batch).detach().max(1)[0][non_terminal_states]
		
		# The expected q-value: E[r + discount_factor * max[a]Q(s', a)]
		expected_q_values = reward_batch + (self.discount_factor * next_state_values)
		
		# Find loss using Huber Loss (smooth l1 loss is Huber loss with delta = 1)
		loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
		
		# Optimize the model
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.agent.dqn.parameters():
			param.grad.data.clamp_(-1, 1)  # Clamp gradient to prevent the exploding gradient problem
		
		self.optimizer.step()
		return loss.item()
	
	
class StateBasedTrainer(Trainer):
	def __init__(
			self,
			agent: Agent,
			environment: Env,
			train_batch_size: int = 128,
			discount_factor: float = 0.999,
			epsilon_start: float = 1.0,
			epsilon_end: float = 0.05,
			epsilon_decay: int = 500,
			target_update: int = 100,
			learning_rate: float = 0.01,
			episodes: int = 250,
			replay_memory_size: int = 10000,
			show_plots: bool = True,
			render_frames: bool = True
	):
		super().__init__(agent, environment, train_batch_size, discount_factor, epsilon_start, epsilon_end,
						epsilon_decay, target_update, learning_rate, episodes, replay_memory_size, show_plots)
		self.render_frames = render_frames
		
	def reset_environment(self):
		self.state = self.environment.reset()
	
	def get_state(self):
		return self.state
	
	def step(self, action):
		state, reward, done, info = self.environment.step(action)
		self.state = state
		if self.render_frames:
			self.environment.render()
		return state, reward, done, info


class VisualTrainer(Trainer):
	def __init__(
			self,
			agent: Agent,
			environment: Env,
			render_size: Tuple[int, int],
			train_batch_size: int = 128,
			discount_factor: float = 0.999,
			epsilon_start: float = 1.0,
			epsilon_end: float = 0.05,
			epsilon_decay: int = 500,
			target_update: int = 100,
			learning_rate: float = 0.01,
			episodes: int = 250,
			replay_memory_size: int = 10000,
			show_plots: bool = True,
	):
		super().__init__(agent, environment, train_batch_size, discount_factor, epsilon_start, epsilon_end,
						epsilon_decay, target_update, learning_rate, episodes, replay_memory_size, show_plots)
		self.render_width = render_size[0]
		self.render_height = render_size[1]
	
	def reset_environment(self):
		self.environment.reset()
		rgb_array = self.environment.render(mode="rgb_array")
		self.state = self.__convert_rgb_array(rgb_array)
		self.state = np.array([self.state]) / 255
		
	def get_state(self):
		return self.state

	def step(self, action):
		observation, reward, done, info = self.environment.step(action)
		rgb_array = self.environment.render(mode="rgb_array")
		self.state = self.__convert_rgb_array(rgb_array)
		self.state = np.array([self.state]) / 255
		return observation, reward, done, info

	def __convert_rgb_array(self, rgb_array):
		gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
		scaled = cv2.resize(gray, (self.render_width, self.render_height))
		return scaled
		