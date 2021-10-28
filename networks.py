import torch
import torch.nn as nn
import torch.nn.functional as F


def output_size(size, kernel_size, stride):
	return (size - (kernel_size - 1) - 1) // stride + 1


class ConvolutionalNetwork(nn.Module):
	def __init__(self, channels, width, height, outputs):
		nn.Module.__init__(self)
		
		conv_out_channels = 64
		
		linear_width = output_size(width, 8, 2)
		linear_width = output_size(linear_width, 2, 1)
		linear_width = output_size(linear_width, 4, 2)
		linear_width = output_size(linear_width, 2, 1)
		linear_width = output_size(linear_width, 4, 2)
		
		linear_height = output_size(height, 8, 2)
		linear_height = output_size(linear_height, 2, 1)
		linear_height = output_size(linear_height, 4, 2)
		linear_height = output_size(linear_height, 2, 1)
		linear_height = output_size(linear_height, 4, 2)
		
		linear_input_size = linear_width * linear_height * conv_out_channels
		
		self.conv1 = nn.Conv2d(channels, 16, kernel_size=8, stride=2)
		self.pool1 = nn.MaxPool2d(2, 1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
		self.pool2 = nn.MaxPool2d(2, 1)
		self.conv3 = nn.Conv2d(32, conv_out_channels, kernel_size=4, stride=2)
		
		self.linear1 = nn.Linear(linear_input_size, 128)
		self.linear2 = nn.Linear(128, 64)
		self.linear3 = nn.Linear(64, outputs)
	
	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x)))
		x = self.pool2(F.relu(self.conv2(x)))
		x = torch.flatten(F.relu(self.conv3(x)), 1)
		x = F.relu(self.linear1(x))
		x = F.relu(self.linear2(x))
		x = self.linear3(x)
		return x


class LinearNetwork(nn.Module):
	def __init__(self, inputs, outputs):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(inputs, 256)
		self.l2 = nn.Linear(256, outputs)
	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = self.l2(x)
		return x
