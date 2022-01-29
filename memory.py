import random
import numpy as np
from collections import deque
from sum_tree import SumTree


class ReplayMemory:
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, transition):
		self.memory.append(transition)

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
	
	def max_size(self):
		return self.memory.maxlen
	

class PERMemory:
	def __init__(self, capacity: int, train_episodes: int, alpha: float = 0.6, beta_start: float = 0.4):
		self.memory = SumTree(capacity)
		self.alpha = alpha
		self.beta = beta_start
		self.beta_increment = (1 - beta_start) / train_episodes
		self.e = 0.01
		
	def push(self, transition):
		value = self.memory.get_max_leaf_value()
		if value == 0:
			value = 1
		self.memory.add(transition, value)
	
	def step_episode(self):
		self.beta = min(1.0, self.beta + self.beta_increment)
	
	def sample(self, batch_size):
		segment_size = self.memory.get_total() / batch_size
		
		samples = []
		sample_indexes = []
		sample_priorities = []
		
		for i in range(batch_size):
			start = segment_size * i
			end = segment_size * (i + 1)
			select = random.uniform(start, end)
			
			index, transition, priority = self.memory.get(select)
			samples.append(transition)
			sample_indexes.append(index)
			sample_priorities.append(priority)
	
		# Normalize the priority to find the importance-sampling weight
		sample_probabilities = sample_priorities / self.memory.get_total()
		is_weights = np.power(len(self.memory) * sample_probabilities, -self.beta)
		is_weights /= is_weights.max()
		
		return samples, sample_indexes, is_weights
	
	def update(self, index, error):
		self.memory.update_value(index, self._calc_priority(error))
		
	def max_size(self):
		return self.memory.capacity
		
	def __len__(self):
		return len(self.memory)
	
	def _calc_priority(self, error):
		return np.abs(error + self.e) ** self.alpha
	