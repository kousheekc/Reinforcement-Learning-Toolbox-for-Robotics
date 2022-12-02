import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Plotter:
	def __init__(self):
		sns.set()
		self.cur_dir = os.getcwd()
		self.plt_dir = os.path.join(self.cur_dir, 'experiments', 'plots')
		if not os.path.exists(self.plt_dir):
			os.makedirs(self.plt_dir)

	def plot_individual(self, title, xlabel, ylabel, algo, env, seed, display=False):
		log_file = os.path.join(self.cur_dir, 'experiments', 'logs', f'{algo}_{env}_{seed}', 'returns.csv')
		log = pd.read_csv(log_file)
		x = list(log['iteration'])
		ep_returns = list(log['avg_return'])

		# Plot
		# ep_returns = pd.Series(ep_returns).rolling(10, min_periods=1).mean()
		ax = sns.lineplot(x=x, y=ep_returns)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)

		filename = os.path.join(self.plt_dir, f'{algo}_{env}_plt.png')
		plt.savefig(filename)

		if display:
			plt.show()

	def combine_csv_files(self, algo, env, log_file_type):
		# Combine all csv files with different seeds into x and y so we can plot
		x = []
		y = []
		num_seeds = 10

		if log_file_type == 'returns':
			for seed in range(num_seeds):
				log_file = os.path.join('experiments', 'logs', f'{algo}_{env}_{seed}', f'{log_file_type}.csv')
				log = pd.read_csv(log_file)
				returns = list(log['avg_return'])
				x += list(log['iteration'])
				y += returns
		elif log_file_type == 'loss':
			for seed in range(num_seeds):
				filename = os.path.join('experiments', 'logs', f'{algo}_{env}_{seed}', f'{log_file_type}.csv')
				df = pd.read_csv(filename)
				col = list(df['loss'])
				x += list(df['iteration'])
				y += col

		return x, y

	def plot_combined(self, title, xlabel, ylabel, algo, env, display=False, log_file_type='returns'):
		"""
		:param log_file_type: Either transitions or policy_updates
		"""
		filename = os.path.join(self.plt_dir, 'all_seeds', algo)
		if not os.path.exists(filename):
			os.makedirs(filename)

		x, y = self.combine_csv_files(algo, env, log_file_type)
		# y = pd.Series(y).rolling(5, min_periods=1).mean()

		# Plot
		ax = sns.lineplot(x=x, y=y, ci=95)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)
		plt.legend([algo + '_' + env])
		plt.savefig(f'{filename}/{env}_all_seeds_plt.png')

		# Display
		if display:
			plt.show()


class Plotterv2:
	def __init__(self):
		sns.set()
		self.cur_dir = os.getcwd()
		self.plt_dir = os.path.join(self.cur_dir, 'experiments', 'plots')
		if not os.path.exists(self.plt_dir):
			os.makedirs(self.plt_dir)

	def plot_individual(self, title, xlabel, ylabel, algo, env, seed, display=True):
		file = os.path.join(self.cur_dir, 'experiments', 'logs', f'{algo}_{env}_{seed}', 'returns.csv')
		epoch_returns = self._get_returns_from_file(file)

		epoch_returns = pd.Series(epoch_returns).rolling(10, min_periods=1).mean()
		ax = sns.lineplot(x=list(range(len(epoch_returns))), y=epoch_returns)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)

		filename = os.path.join(self.plt_dir, f'{algo}_{env}_{seed}_plt.png')
		plt.savefig(filename)

		if display:
			plt.show()


	def plot_combined(self, title, xlabel, ylabel, algo, env, num_of_seeds=10, display=True):
		x = []
		y = []

		for seed in range(num_of_seeds):
			file = os.path.join(self.cur_dir, 'experiments', 'logs', f'{algo}_{env}_{seed}', 'returns.csv')
			epoch_returns = self._get_returns_from_file(file)

			x += list(range(len(epoch_returns)))
			y += list(epoch_returns)

		y = pd.Series(y).rolling(5, min_periods=1).mean()

		# Plot
		ax = sns.lineplot(x=x, y=y, ci=95)
		ax.axes.set_title(title, fontsize=20)
		ax.set_xlabel(xlabel, fontsize=15)
		ax.set_ylabel(ylabel, fontsize=15)
		plt.legend([algo + '_' + env], loc='lower right')

		filename = os.path.join(self.plt_dir, 'all_seeds', algo)
		plt.savefig(f'{filename}/{env}_all_seeds_plt.png')

		# Display
		if display:
			plt.show()

	def _get_returns_from_file(self, file):
		epoch_returns = []
		epoch_return_logs = pd.read_csv(file)

		for i in range(len(epoch_return_logs)):
			epoch_returns.append(epoch_return_logs.loc[i, 'returns'])

		return epoch_returns


# p = Plotter()
# p.plot_combined('CartPole-v1 DQN', 'Timesteps', 'Mean Return', 'DQN', 'CartPole-v1', display=True)
