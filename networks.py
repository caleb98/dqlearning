import torch.nn as nn
import torch.nn.functional as F


def conv2d_size_out(size, kernel_size=5, stride=2):
	return (size - (kernel_size - 1) - 1) // stride + 1


def create_dqn_cnn(width, height, outputs):
	convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
	convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
	linear_input_size = convw * convh * 32
	return nn.Sequential(
		nn.Conv2d(3, 16, kernel_size=5, stride=2),
		nn.BatchNorm2d(16),
		nn.ReLU(),
		nn.Conv2d(16, 32, kernel_size=5, stride=2),
		nn.BatchNorm2d(32),
		nn.ReLU(),
		nn.Conv2d(32, 32, kernel_size=5, stride=2),
		nn.BatchNorm2d(32),
		nn.ReLU(),
		nn.Flatten(),
		nn.Linear(linear_input_size, outputs)
	)


class LinearNetwork(nn.Module):
	def __init__(self, inputs, outputs):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(inputs, 256)
		self.l2 = nn.Linear(256, outputs)
	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = self.l2(x)
		return x
