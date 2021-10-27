from torch import nn
from torch import optim

class Agent:
	def __init__(self, network, target, environment, train_batch_size=128, discount_factor=0.999,
				epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=500, target_update=100, learning_rate=0.01,
				episodes=250):
		self.dqn = network
		self.dqn_target = target
		self.env = environment
		
		self.train_episodes = episodes
		self.train_batch_size = train_batch_size
		self.discount_factor = discount_factor
		
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		
		self.target_update = target_update
		self.optimizer = optim.RMSprop(self.dqn.parameters(), lr=learning_rate)
	
