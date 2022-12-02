import numpy as np


class GaussianNoise:
	def __init__(self, seed, act_dim):
		np.random.seed(seed)
		self.act_dim = act_dim

	def __call__(self, noise_scale):
		return noise_scale * np.random.rand(self.act_dim)
