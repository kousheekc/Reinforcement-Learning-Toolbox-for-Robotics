import torch
import itertools
import numpy as np
from copy import deepcopy

from rlbotics.common.loss import losses
from rlbotics.td3.utils import GaussianNoise
from rlbotics.td3.replay_buffer import ReplayBuffer
from rlbotics.common.policies import MLPActorContinuous, MLPQFunctionContinuous


class TD3:
	"""
	NOTE: For Continuous environments only
	"""
	def __init__(self, args, env):
		self.act_dim = env.action_space.shape[0]
		self.obs_dim = env.observation_space.shape[0]
		self.act_lim = env.action_space.high[0]

		# General parameters
		self.env = env
		self.seed = args.seed
		self.gamma = args.gamma
		self.env_name = args.env_name
		self.save_freq = args.save_freq
		self.use_grad_clip = args.use_grad_clip

		# TD3 Specific Parameters
		self.batch_size = args.batch_size
		self.buffer_size = args.buffer_size
		self.polyak = args.polyak
		self.act_noise = args.act_noise
		self.random_steps = args.random_steps
		self.update_after = args.update_after

		# Policy Network Parameters
		self.pi_lr = args.pi_lr
		self.pi_hidden_sizes = args.pi_hidden_sizes
		self.pi_activations = args.pi_activations
		self.pi_optimizer = args.pi_optimizer
		self.pi_update_delay = args.pi_update_delay
		self.pi_targ_noise = args.pi_targ_noise
		self.noise_clip = args.noise_clip

		# Q Network Parameters
		self.q_lr = args.q_lr
		self.q_hidden_sizes = args.q_hidden_sizes
		self.q_activations = args.q_activations
		self.q_optimizer = args.q_optimizer
		self.q_loss_type = args.q_loss_type
		self.weight_decay = args.weight_decay

		# Both networks
		self.weight_init = args.weight_init
		self.batch_norm = args.batch_norm

		# Set device
		gpu = 0
		self.device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")

		# Initialize action noise
		self.noise = GaussianNoise(self.seed, self.act_dim)

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
		self.q_criterion = losses(self.q_loss_type)

		# Build pi and q Networks
		self._build_policy()
		self._build_q_function()

	def _build_policy(self):
		layer_sizes = [self.obs_dim] + self.pi_hidden_sizes + [self.act_dim]
		self.pi = MLPActorContinuous(act_lim=self.act_lim,
									 layer_sizes=layer_sizes,
									 activations=self.pi_activations,
									 seed=self.seed,
									 batch_norm=self.batch_norm,
									 weight_init=self.weight_init).to(self.device)

		self.pi.summary()

		# Set Optimizer
		if self.pi_optimizer == 'Adam':
			self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=self.pi_lr)
		elif self.pi_optimizer == 'RMSprop':
			self.pi_optim = torch.optim.RMSprop(self.pi.parameters(), lr=self.pi_lr)
		else:
			raise NameError(str(self.pi_optimizer) + ' Optimizer not supported')

		# Build Target
		self.pi_target = deepcopy(self.pi).to(self.device)

		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.pi_target.parameters():
			p.requires_grad = False

	def _build_q_function(self):
		layer_sizes = [self.obs_dim + self.act_dim] + self.q_hidden_sizes + [1]
		self.q1 = MLPQFunctionContinuous(layer_sizes=layer_sizes,
										 activations=self.q_activations,
										 seed=self.seed,
										 batch_norm=self.batch_norm,
										 weight_init=self.weight_init).to(self.device)

		self.q2 = MLPQFunctionContinuous(layer_sizes=layer_sizes,
										 activations=self.q_activations,
										 seed=self.seed,
										 batch_norm=self.batch_norm,
										 weight_init=self.weight_init).to(self.device)

		self.q1.summary()
		self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())

		# Set Optimizer
		if self.q_optimizer == 'Adam':
			self.q_optim = torch.optim.Adam(self.q_params, lr=self.q_lr, weight_decay=self.weight_decay)
		elif self.q_optimizer == 'RMSprop':
			self.q_optim = torch.optim.RMSprop(self.q_params, lr=self.q_lr, weight_decay=self.weight_decay)
		else:
			raise NameError(str(self.q_optimizer) + ' Optimizer not supported')

		# Build Target
		self.q1_target = deepcopy(self.q1).to(self.device)
		self.q2_target = deepcopy(self.q2).to(self.device)

		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p1, p2 in zip(self.q1_target.parameters(), self.q2_target.parameters()):
			p1.requires_grad = False
			p2.requires_grad = False

	def _compute_q_loss(self, batch):
		obs_batch = torch.as_tensor(batch.obs, dtype=torch.float).to(self.device)
		act_batch = torch.as_tensor(batch.act).to(self.device)
		rew_batch = torch.as_tensor(batch.rew).unsqueeze(1).to(self.device)
		new_obs_batch = torch.as_tensor(batch.new_obs, dtype=torch.float).to(self.device)
		done_batch = torch.as_tensor(batch.done).to(self.device)
		not_done_batch = torch.logical_not(done_batch).unsqueeze(1).to(self.device)

		q1_pred = self.q1.get_q_value(obs_batch, act_batch)
		q2_pred = self.q2.get_q_value(obs_batch, act_batch)

		# Bellman backup for Q function
		with torch.no_grad():
			pi_targ = self.pi.get_action(new_obs_batch)

			# Target policy smoothing
			noise = torch.randn_like(pi_targ) * self.noise(self.pi_targ_noise)
			noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
			pi_targ += noise
			pi_targ = torch.clamp(pi_targ, -self.act_lim, self.act_lim).float()

			# Target Q-values
			q1_pi_targ = self.q1_target.get_q_value(new_obs_batch, pi_targ)
			q2_pi_targ = self.q2_target.get_q_value(new_obs_batch, pi_targ)
			q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
			expected_q = rew_batch + self.gamma * (q_pi_targ * not_done_batch)

		q1_loss = self.q_criterion(q1_pred, expected_q.float())
		q2_loss = self.q_criterion(q2_pred, expected_q.float())
		loss = q1_loss + q2_loss
		return loss

	def _compute_pi_loss(self, batch):
		obs_batch = torch.as_tensor(batch.obs, dtype=torch.float).to(self.device)
		pi = self.pi.get_action(obs_batch)
		q = self.q1.get_q_value(obs_batch, pi)
		return -q.mean()

	def update_actor_critic(self):
		if self.iteration < self.update_after or len(self.memory) < self.batch_size:
			return

		# Sample batch of transitions
		transition_batch = self.memory.sample(self.batch_size)

		# Update q Network
		q_loss = self._compute_q_loss(transition_batch)
		self.q_optim.zero_grad()
		q_loss.backward()
		if self.grad_clip is not None:
			for param in self.q_params:
				param.grad.data.clamp_(self.grad_clip[0], self.grad_clip[1])
		self.q_optim.step()

		# If it's time to update policy and target networks
		if self.iteration % self.pi_update_delay == 0:
			# Freeze Q-network so you don't waste computational effort
			# computing gradients for it during the policy learning step.
			for p in self.q_params:
				p.requires_grad = False

			# Update pi Network
			pi_loss = self._compute_pi_loss(transition_batch)
			self.pi_optim.zero_grad()
			pi_loss.backward()
			if self.grad_clip is not None:
				for param in self.pi.parameters():
					param.grad.data.clamp_(self.grad_clip[0], self.grad_clip[1])
			self.pi_optim.step()

			# Unfreeze Q-network so you can optimize it at next TD3 step.
			for p in self.q_params:
				p.requires_grad = True

			# Update Target Networks
			self._update_target_actor_critic()

			return q_loss.item(), pi_loss.item()

	def _update_target_actor_critic(self):
		# Polyak averaging
		with torch.no_grad():
			for p1, p2, p1_targ, p2_targ in zip(self.q1.parameters(), self.q2.parameters(), self.q1_target.parameters(), self.q2_target.parameters()):
				p1_targ.data.copy_(self.polyak*p1.data + (1-self.polyak)*p1_targ.data)
				p2_targ.data.copy_(self.polyak*p2.data + (1-self.polyak)*p2_targ.data)
			for p, p_targ in zip(self.pi.parameters(), self.pi_target.parameters()):
				p_targ.data.copy_(self.polyak*p.data + (1-self.polyak)*p_targ.data)

	def get_action(self, obs):
		self.pi.eval()
		action = self.pi.get_action(obs).detach().numpy()
		action += self.noise(self.act_noise)
		return np.clip(action, -self.act_lim, self.act_lim)[0]

	def store_transition(self, obs, act, rew, new_obs, done):
		self.memory.add(obs, act, rew, new_obs, done)
		self.iteration += 1
