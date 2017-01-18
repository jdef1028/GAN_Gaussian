'''
This is an implementation of Genarative Adversarial Networks in TensorFlow

Xiaolin Li

Project started on Jan 17, 2016
'''
import numpy as np

class DataDistribution(object):
	def __init__(self):
		# assign a normal distribution N(mu, sigma) for the data distribution
		self.mu = 4
		self.sigma = 0.5

	def sample(self, N):
		# draw N samples from the data distribution
		samples = np.random.normal(self.mu, self.sigma, N)
		samples.sort()

		return samples


class GeneratorDistribution(object):
	def __init__(self, range):
		self.range = range

	def sample(self, N):
		# sample N samples from the generator distribution (some noise is added)
		return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.02


