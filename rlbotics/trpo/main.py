import gym
import torch
import argparse

from rlbotics.trpo.trpo import TRPO
import rlbotics.trpo.hyperparameters as h
from rlbotics.common.plotter import Plotter


def argparser():
	"""
	Input argument parser.
	Loads default hyperparameters from hyperparameters.py
	:return: Parsed arguments
	"""
	parser = argparse.ArgumentParser()
	# General Parameters
	parser.add_argument('--seed', type=int, default=h.seed)
	parser.add_argument('--env_name', type=str, default=h.env_name)
	parser.add_argument('--gamma', type=float, default=h.gamma)
	parser.add_argument('--lam', type=float, default=h.lam)
	parser.add_argument('--max_iterations', type=int, default=h.max_iterations)
	parser.add_argument('--max_epochs', type=int, default=h.max_epochs)
	parser.add_argument('--render', type=bool, default=h.render)
	parser.add_argument('--num_v_iters', type=int, default=h.num_v_iters)

	# TRPO specific hyperparameters
	parser.add_argument('--kl_target', type=float, default=h.kl_target)

	# Policy Network:
	parser.add_argument('--pi_hidden_sizes', nargs='+', type=int, default=h.pi_hidden_sizes)
	parser.add_argument('--pi_activations', nargs='+', type=str, default=h.pi_activations)
	parser.add_argument('--pi_lr', type=float, default=h.pi_lr)
	parser.add_argument('--pi_optimizer', type=str, default=h.pi_optimizer)

	# Value Network:
	parser.add_argument('--v_hidden_sizes', nargs='+', type=int, default=h.v_hidden_sizes)
	parser.add_argument('--v_activations', nargs='+', type=str, default=h.v_activations)
	parser.add_argument('--v_lr', type=float, default=h.v_lr)
	parser.add_argument('--v_optimizer', type=str, default=h.v_optimizer)

	return parser.parse_args()

def main():
	args = argparser()

	# Set device
	gpu = 0
	device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")
	print(device)
	if torch.cuda.is_available():
		torch.cuda.set_device(device)

	# Build environment
	env = gym.make(args.env_name)
	env.seed(args.seed)
	agent = TRPO(args, env)
	obs = env.reset()

	# Episode related information
	ep_counter = 0
	ep_rew = 0

	for epoch in range(args.max_epochs):
		for iteration in range(args.max_iterations):
			if h.render:
				env.render()

			# Take action
			act = agent.get_action(obs)
			new_obs, rew, done, _ = env.step(act)

			# Store experience
			agent.store_transition(obs, act, rew, new_obs, done)

			ep_rew += rew
			obs = new_obs

			# Episode done
			if done:
				obs = env.reset()
				# Display results
				print("epoch: {}, episode: {}, total reward: {}".format(epoch, ep_counter, ep_rew))

				# Logging
				ep_counter += 1
				ep_rew = 0

		# Update Policy
		agent.update_policy()

		# Update Value
		agent.update_value()

	# End
	env.close()
	p = Plotter()
	p.plot_individual('Epoch/Reward', 'epochs', 'rewards', 'TRPO', args.env_name, args.seed, epoch_iter=args.max_iterations, display=True)


if __name__ == '__main__':
	main()
