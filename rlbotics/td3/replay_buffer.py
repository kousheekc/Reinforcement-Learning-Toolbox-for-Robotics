from collections import namedtuple, deque
import random


class ReplayBuffer:
	def __init__(self, buffer_size, seed):
		random.seed(seed)
		self.memory = deque(maxlen=int(buffer_size))
		self.transition = namedtuple("transition", field_names=["obs", "act", "rew", "new_obs", "done"])

	def add(self, obs, act, rew, new_obs, done):
		transition = self.transition(obs=obs, act=act, rew=rew, new_obs=new_obs, done=done)
		self.memory.append(transition)

	def sample(self, batch_size):
		transition_batch = random.sample(self.memory, batch_size)

		# Transpose transitions
		transition_batch = self.transition(*zip(*transition_batch))
		return transition_batch

	def __len__(self):
		return len(self.memory)
