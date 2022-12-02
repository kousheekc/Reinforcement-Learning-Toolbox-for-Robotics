import numpy as np


class OUNoise:
	def __init__(self, seed, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
		"""
		Ornstein-Uhlenbeck action noise
		:param mu: mean
		:param sigma: standard deviation
		"""
		np.random.seed(seed)
		self.mu = mu
		self.sigma = sigma
		self.theta = theta
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = (
			self.x_prev
			+ self.theta * (self.mu - self.x_prev) * self.dt
			+ self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		)
		self.x_prev = x
		return x

	def reset(self):
		if self.x0 is not None:
			self.x_prev = self.x0
		else:
			self.x_prev = np.zeros_like(self.mu)


class GaussianNoise:
	def __init__(self, seed, act_noise, act_dim):
		np.random.seed(seed)
		self.noise_scale = act_noise
		self.act_dim = act_dim

	def __call__(self):
		return self.noise_scale * np.random.rand(self.act_dim)
