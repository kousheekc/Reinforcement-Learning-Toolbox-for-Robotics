import gym
import torch
import argparse

from rlbotics.vpg.vpg import VPG
import rlbotics.vpg.hyperparameters as h
from rlbotics.common.logger import Loggerv2
from rlbotics.common.plotter import Plotterv2


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
	parser.add_argument('--memory_size', type=int, default=h.memory_size)

	# Policy Network:
	parser.add_argument('--pi_lr', type=float, default=h.pi_lr)
	parser.add_argument('--pi_hidden_sizes', nargs='+', type=int, default=h.pi_hidden_sizes)
	parser.add_argument('--pi_activations', nargs='+', type=str, default=h.pi_activations)
	parser.add_argument('--pi_optimizer', type=str, default=h.pi_optimizer)

	# Value Network:
	parser.add_argument('--v_lr', type=float, default=h.v_lr)
	parser.add_argument('--v_hidden_sizes', nargs='+', type=int, default=h.v_hidden_sizes)
	parser.add_argument('--v_activations', nargs='+', type=str, default=h.v_activations)
	parser.add_argument('--num_v_iters', type=int, default=h.num_v_iters)
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
	agent = VPG(args, env)

	logs = Loggerv2('VPG', args.env_name, args.seed)

	obs = env.reset()

	# Episode related information
	ep_counter = 0
	ep_rew = 0

	for epoch in range(args.max_epochs):
		for iteration in range(args.max_iterations):
			if h.render:
				env.render()

			# Take action
			act = agent.policy.get_action(torch.as_tensor(obs, dtype=torch.float32))

			new_obs, rew, done, _ = env.step(act)
			ep_rew += rew

			# Store experience
			agent.store_transition(obs, act, rew)

			obs = new_obs

			epoch_ended = iteration==args.max_iterations-1

			# Episode done or epoch terminated while episode not yet complete
			if done or epoch_ended:
				# calculate value of current state to estimate return if trajectory was cut halfway, else value = 0
				if epoch_ended:
					v = agent.value(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
				else:
					v = 0
				agent.memory.finish_path(v)
				if done:
					# log ony if episode complete
					# print("total reward: {}".format(ep_ret))
					logs.log_return(ep_rew)

				obs = env.reset()
				ep_rew = 0

		# Update Policy
		loss_pi = agent.update_policy()
		# Update Value
		loss_val = agent.update_value()

		logs.log_policy(loss_pi=loss_pi, loss_val=loss_val)

		epoch_return = logs.log_return()
		print("epoch: {}, average return: {}".format(epoch, epoch_return))

	# End
	env.close()
	p = Plotterv2()
	p.plot_individual('Mean return/Epoch', 'Epoch', 'Mean return', 'VPG', args.env_name, args.seed)


if __name__ == '__main__':
	main()
