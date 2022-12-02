import math
import torch
from copy import deepcopy

from rlbotics.common.loss import losses
from rlbotics.ddqn.replay_buffer import ReplayBuffer
from rlbotics.common.policies import MLPEpsilonGreedy


class DDQN:
	def __init__(self, args, env):
		self.env_name = args.env_name
		self.act_dim = env.action_space.n
		self.obs_dim = env.observation_space.shape[0]

		# General parameters
		self.lr = args.lr
		self.seed = args.seed
		self.gamma = args.gamma
		self.save_freq = args.save_freq

		# DDQN specific parameters
		self.epsilon = args.epsilon
		self.min_epsilon = args.min_epsilon
		self.exp_decay = args.exp_decay
		self.linear_decay = args.linear_decay
		self.batch_size = args.batch_size
		self.buffer_size = args.buffer_size
		self.update_target_freq = args.update_target_freq

		# Policy network parameters
		self.loss_type = args.loss_type
		self.optimizer = args.optimizer
		self.use_grad_clip = args.use_grad_clip
		self.activations = args.activations
		self.hidden_sizes = args.hidden_sizes

		# Set device
		gpu = 0
		self.device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")

		# Replay buffer
		self.memory = ReplayBuffer(self.buffer_size, self.seed)

		# Gradient clipping
		if self.use_grad_clip:
			self.grad_clip = (-1, 1)
		else:
			self.grad_clip = None

		# Steps
		self.iteration = 0

		# Loss function
		self.criterion = losses(self.loss_type)

		# Build policies
		self._build_policy()

	def _build_policy(self):
		layer_sizes = [self.obs_dim] + self.hidden_sizes + [self.act_dim]
		self.policy = MLPEpsilonGreedy(layer_sizes=layer_sizes,
									   activations=self.activations,
									   seed=self.seed).to(self.device)

		self.target_policy = deepcopy(self.policy).to(self.device)
		self.policy.summary()

		# Set Optimizer
		if self.optimizer == 'Adam':
			self.optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
		elif self.optimizer == 'RMSprop':
			self.q_optim = torch.optim.RMSprop(self.policy.parameters(), lr=self.lr)
		else:
			raise NameError(str(self.optimizer) + ' Optimizer not supported')

	def get_action(self, obs):
		action = self.policy.get_action(obs, self.epsilon)
		self.decay_epsilon(mode='exp')
		return action

	def decay_epsilon(self, mode):
		if mode == 'exp':
			self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * math.exp(-self.exp_decay * self.iteration)
			self.iteration += 1
		elif mode == 'linear':
			self.epsilon = max(self.min_epsilon, self.epsilon-self.linear_decay)

	def store_transition(self, obs, act, rew, new_obs, done):
		self.memory.add(obs, act, rew, new_obs, done)

	def update_policy(self):
		if len(self.memory) < self.batch_size:
			return

		# Sample batch of transitions
		transition_batch = self.memory.sample(self.batch_size)

		# Extract batches and convert to tensors
		obs_batch = torch.as_tensor(transition_batch.obs, dtype=torch.float).to(self.device)
		act_batch = torch.as_tensor(transition_batch.act).to(self.device)
		rew_batch = torch.as_tensor(transition_batch.rew).to(self.device)
		new_obs_batch = torch.as_tensor(transition_batch.new_obs, dtype=torch.float).to(self.device)
		done_batch = torch.as_tensor(transition_batch.done).to(self.device)
		not_done_batch = torch.logical_not(done_batch).to(self.device)

		# Update
		q_values = self.policy(obs_batch).gather(1, act_batch.unsqueeze(1))
		next_state_q_values = self.policy(new_obs_batch[not_done_batch]).argmax(1)
		target_values = torch.zeros(self.batch_size, 1).to(self.device)
		target_values[not_done_batch] = self.target_policy(new_obs_batch[not_done_batch]).gather(1, next_state_q_values.unsqueeze(1)).detach()

		expected_q_values = rew_batch.unsqueeze(1) + self.gamma * target_values

		loss = self.criterion(q_values, expected_q_values.float()).to(self.device)

		# Learn
		self.optim.zero_grad()
		loss.backward()
		if self.grad_clip is not None:
			for param in self.policy.parameters():
				param.grad.data.clamp_(self.grad_clip[0], self.grad_clip[1])
		self.optim.step()

		# Update target policy
		if self.iteration % self.update_target_freq == 0:
			self.update_target_policy()

		return loss.item()

	def update_target_policy(self):
		self.target_policy.load_state_dict(self.policy.state_dict())
