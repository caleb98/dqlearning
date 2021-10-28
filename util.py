import random
from collections import deque


class ReplayMemory(object):
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
