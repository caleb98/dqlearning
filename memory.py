"""
Contains standard and PER-based replay memory implementations.
"""

import random
import numpy as np
from collections import deque
from sum_tree import SumTree


class ReplayMemory:
	"""A standard replay memory which stores a collection of transitions."""
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)

	def push(self, transition):
		"""Adds a new transition to the replay memory. Overwrites oldest transition if the capacity is exceeded."""
		self.memory.append(transition)

	def sample(self, batch_size):
		"""Samples a random collection of transitions from the replay memory based on the requested batch size."""
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
	
	def max_size(self):
		"""Returns the maximum number of transitions that can be stored in this replay memory."""
		return self.memory.maxlen
	

class PERMemory:
	"""An implementation of PER-based replay memory."""
	
	def __init__(self, capacity: int, train_episodes: int, alpha: float = 0.6, beta_start: float = 0.4):
		"""
		:param capacity: maximum transitions to store
		:param train_episodes: number of training epsidoes that this replay memory will be used for
		:param alpha: alpha value for priority calculation
		:param beta_start: starting beta value for importance-sampling weight calculation (annealed to 1 over the course of training)
		"""
		self.memory = SumTree(capacity)
		self.alpha = alpha
		self.beta = beta_start
		self.beta_increment = (1 - beta_start) / train_episodes
		self.e = 0.0001
		
	def push(self, transition):
		"""Adds a new transition to the replay memory. Overwrites oldest transition if the capacity is exceeded."""
		value = self.memory.get_max_leaf_value()
		if value == 0:
			value = 1
		self.memory.add(transition, value)
	
	def step_episode(self):
		"""Updates the beta value used for importance-sampling weights calculation."""
		self.beta = min(1.0, self.beta + self.beta_increment)
	
	def sample(self, batch_size):
		"""Samples a random collection of transitions from the replay memory based on the priority of each transition."""
		# Split the memory into equal-size segments based on transition priorities
		segment_size = self.memory.get_total() / batch_size
		
		samples = []
		sample_indexes = []
		sample_priorities = []
		
		# Loop through each segment
		for i in range(batch_size):
			# Calculate the segment boundaries
			start = segment_size * i
			end = segment_size * (i + 1)
			
			# Select a random position within the segment
			select = random.uniform(start, end)
			
			# Pull the transition that is at that position, and add it to the list of samples
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
		"""Updates the priority of a transition at a specific index based on the error in the prediction."""
		self.memory.update_value(index, self._calc_priority(error))
		
	def max_size(self):
		"""Returns the maximum number of transitions that can be stored in this replay memory."""
		return self.memory.capacity
		
	def __len__(self):
		return len(self.memory)
	
	def _calc_priority(self, error):
		"""Calculates the priority of a transition based on the error in the prediction."""
		return np.abs(error + self.e) ** self.alpha
	