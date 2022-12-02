import os
import csv
import json
import torch
import shutil
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger:
	def __init__(self, algo_name, env_name, seed, resume=False):
		"""
		:param algo_name: (str) name of file in which log will be stored
		:param env_name: (str) name of environment used in experiment
		:param seed: (int) random seed used in experiment
		:param resume: (bool) Resume training some model
		"""
		cur_dir = os.getcwd()
		self.log_dir = os.path.join(cur_dir, 'experiments', 'logs', f'{algo_name}_{env_name}_{seed}')
		self.model_dir = os.path.join(cur_dir, 'experiments', 'models', f'{algo_name}_{env_name}_{seed}')
		if os.path.exists(self.log_dir) and not resume:
			shutil.rmtree(self.log_dir)
		if os.path.exists(self.model_dir) and not resume:
			shutil.rmtree(self.model_dir)
		if not resume:
			os.makedirs(self.log_dir)
			os.makedirs(self.model_dir)

		self.resume = resume
		self.returns_header, self.policy_updates_header = True, True

		# Tensor Board
		self.writer = SummaryWriter(log_dir=self.log_dir)

		if self.resume:
			self.resume_log()

	def log_return(self, iteration, avg_return):
		file = os.path.join(self.log_dir, 'returns.csv')
		self._save_tabular(file, self.returns_header, iteration=iteration, avg_return=avg_return)
		self.returns_header = False

	def log_policy_update(self, iteration, **kwargs):
		file = os.path.join(self.log_dir, 'policy_updates.csv')
		self._save_tabular(file, self.policy_updates_header, iteration=iteration, **kwargs)
		self.policy_updates_header = False

	def log_params(self, **kwargs):
		file = os.path.join(self.log_dir, 'params.json')
		with open(file, 'w') as f:
			json.dump(kwargs, f, indent=4)

	def _save_tabular(self, file, header, **kwargs):
		with open(file, 'a') as f:
			writer = csv.writer(f)
			if header:
				writer.writerow(kwargs.keys())
			writer.writerow(kwargs.values())

		self._write_tensorboard(file, **kwargs)

	def _write_tensorboard(self, file, **kwargs):
		if 'returns.csv' in file:
			self.writer.add_scalar('mean reward/timestep', kwargs['avg_return'], kwargs['iteration'])
		else:
			iteration = kwargs['iteration']
			kwargs.pop('iteration')
			for key, value in kwargs.items():
				self.writer.add_scalar(key, value, iteration)

	def log_model(self, mlp, name=''):
		file = os.path.join(self.model_dir, name + 'model.pth')
		torch.save(mlp, file)

	def log_state_dict(self, dct, name):
		file = os.path.join(self.model_dir, name)
		torch.save(dct, file)

	def resume_log(self):
		raise NotImplementedError


class Loggerv2:
	def __init__(self, algo_name, env_name, seed, resume=False):
		"""
		:param algo_name: (str) name of file in which log will be stored
		:param env_name: (str) name of environment used in experiment
		:param seed: (int) random seed used in experiment
		:param resume: (bool) Resume training some model
		"""
		cur_dir = os.getcwd()
		self.log_dir = os.path.join(cur_dir, 'experiments', 'logs', f'{algo_name}_{env_name}_{seed}')
		self.model_dir = os.path.join(cur_dir, 'experiments', 'models', f'{algo_name}_{env_name}_{seed}')
		if os.path.exists(self.log_dir) and not resume:
			shutil.rmtree(self.log_dir)
		if os.path.exists(self.model_dir) and not resume:
			shutil.rmtree(self.model_dir)
		if not resume:
			os.makedirs(self.log_dir)
			os.makedirs(self.model_dir)

		self.resume = resume
		self.transition_keys, self.policy_update_keys = [], []

		# Tensor Board
		self.writer = SummaryWriter(log_dir=self.log_dir)

		# Counter to track number of policy updates
		self.tensorboard_updated_returns = 0
		self.tensorboard_updated_policy = 0

		# Keeps track of returns from episodes for each epoch
		self.episode_returns = []

		if self.resume:
			self.resume_log()

	def log_return(self, ep_ret=None):
		if ep_ret == None:
			file = os.path.join(self.log_dir, 'returns.csv')
			header = True if len(self.transition_keys) == 0 and not self.resume else False
			mean_return_epoch = np.mean(self.episode_returns)
			self.episode_returns.clear()

			with open(file, 'a') as f:
				writer = csv.writer(f)
				if header:
					writer.writerow(['returns'])
					self.transition_keys = ['returns']
				writer.writerow([mean_return_epoch])

			self.writer.add_scalar('mean reward/epoch', mean_return_epoch, self.tensorboard_updated_returns)
			self.tensorboard_updated_returns += 1
			return mean_return_epoch
		else:
			self.episode_returns.append(ep_ret)
			return None

	def log_policy(self, **kwargs):
		file = os.path.join(self.log_dir, 'policy_updates.csv')
		header = True if len(self.policy_update_keys) == 0 and not self.resume else False
		if header:
			self.policy_update_keys = list(kwargs.keys())
		self._save_tabular(file, header, **kwargs)

	def log_params(self, **kwargs):
		file = os.path.join(self.log_dir, 'params.json')
		with open(file, 'w') as f:
			json.dump(kwargs, f, indent=4)

	def _save_tabular(self, file, header, **kwargs):
		with open(file, 'a') as f:
			writer = csv.writer(f)
			if header:
				writer.writerow(kwargs.keys())
			writer.writerow(kwargs.values())

		for key, value in kwargs.items():
			self.writer.add_scalar(key, value, self.tensorboard_updated_policy)

		self.tensorboard_updated_policy += 1

	def log_model(self, mlp, name=''):
		file = os.path.join(self.model_dir, name + 'model.pth')
		torch.save(mlp, file)

	def log_state_dict(self, dct, name):
		file = os.path.join(self.model_dir, name)
		torch.save(dct, file)

	def resume_log(self):
		log_file = os.path.join(self.log_dir, 'returns.csv')
		log = pd.read_csv(log_file)
		for i in range(len(log)):
			if log.loc[i, 'done']:
				self.tensorboard_updated += 1
