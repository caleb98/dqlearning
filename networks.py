"""
Contains the networks used by environment agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNetwork(nn.Module):
	"""A simple linear network consisting of a single hidden layer with 256 neurons."""
	def __init__(self, inputs, outputs):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(inputs, 256)
		self.l2 = nn.Linear(256, outputs)
	
	def forward(self, x):
		"""Takes the given input and uses it to do a pass of the network. Returns the output layer values."""
		x = torch.flatten(x, 1)
		x = F.relu(self.l1(x))
		x = self.l2(x)
		return x
