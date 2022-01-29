import numpy


# Implementation of the "sum tree" data structure described in Schaul et. al 2015.
# This is effectively a binary heap, except nodes are not sorted in any specific
# order. Instead, each parent node expresses the sum value of all its children.
class SumTree:
	
	def __init__(self, capacity):
		self.capacity = capacity
		self.size = 0
		self.insert = 0
		self.values = numpy.zeros(2 * capacity - 1)
		self.elements = numpy.zeros(capacity, dtype=tuple)
	
	# Returns the max value of all the leaf nodes
	def get_max_leaf_value(self):
		start = self.capacity - 1
		return self.values[start:].max()
	
	def get_total(self):
		return self.values[0]
	
	def get(self, select_value):
		if select_value > self.get_total():
			return -1, None, None
		
		# Move through the tree starting from the root node
		index = 0
		while index < self.capacity - 1:
			# Get the left and right nodes of the current nodes
			left = 2 * index + 1
			right = left + 1
			
			# In rare cases, floating point inaccuracies can lead to problems
			# where the "select_value" is greater than the total value of all
			# children nodes. To fix that, we just do a quick check for that
			# condition here, and if that *is* the case, we nudge the value
			# back into the proper range.
			if select_value > self.values[left] + self.values[right]:
				select_value = (self.values[left] + self.values[right]) * 0.99
			
			# If less than left node, move left
			if select_value <= self.values[left]:
				index = left
			
			# Otherwise, move right and adjust search value
			else:
				index = right
				select_value -= self.values[left]
			
		# Return once the index has reached a leaf node
		element_index = index - (self.capacity - 1)
		return index, self.elements[element_index], self.values[index]
	
	def add(self, data, value):
		# Append data to elements array
		self.elements[self.insert] = data
		
		# Insert the element value into the tree and update
		tree_index = self.insert + (self.capacity - 1)
		self.update_value(tree_index, value)
		
		# Loop insertion back around if we've maxed capacity
		self.insert += 1
		if self.insert == self.capacity:
			self.insert = 0

		if self.size < self.capacity:
			self.size += 1
	
	def update_value(self, index, new_value):
		# Set the value at the index
		change = new_value - self.values[index]
		self.values[index] = new_value
		
		# Keep moving up the tree and update intermediate
		# nodes until the root node (0) is updated.
		while index != 0:
			index = (index - 1) // 2
			self.values[index] += change
		
	def __len__(self):
		return self.size
		
	