import gym
import argparse

from rlbotics.dqn.dqn import DQN
import rlbotics.dqn.hyperparameters as h
from rlbotics.common.logger import Logger
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
	parser.add_argument('--lr', type=float, default=h.lr)
	parser.add_argument('--max_iterations', type=int, default=h.max_iterations)
	parser.add_argument('--eval_freq', type=int, default=h.eval_freq)
	parser.add_argument('--render', type=int, default=h.render)
	parser.add_argument('--use_grad_clip', type=int, default=h.use_grad_clip)
	parser.add_argument('--save_freq', type=int, default=h.save_freq)

	# DQN Specific Parameters
	parser.add_argument('--batch_size', type=int, default=h.batch_size)
	parser.add_argument('--buffer_size', type=int, default=h.buffer_size)
	parser.add_argument('--epsilon', type=float, default=h.epsilon)
	parser.add_argument('--min_epsilon', type=float, default=h.min_epsilon)
	parser.add_argument('--exp_decay', type=float, default=h.exp_decay)
	parser.add_argument('--linear_decay', type=float, default=h.linear_decay)

	# Policy/Target Network
	parser.add_argument('--hidden_sizes', nargs='+', type=int, default=h.hidden_sizes)
	parser.add_argument('--activations', nargs='+', type=str, default=h.activations)
	parser.add_argument('--optimizer', type=str, default=h.optimizer)
	parser.add_argument('--loss_type', type=str, default=h.loss_type)
	parser.add_argument('--update_target_freq', type=int, default=h.update_target_freq)

	return parser.parse_args()


def eval_policy(policy, env_name, seed, eval_episodes=10):
	"""
	Evaluate policy over certain number of episodes (no exploration or noise)
	"""
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_rew = 0.
	for _ in range(eval_episodes):
		obs, done = eval_env.reset(), False
		while not done:
			act = policy.get_action(obs, epsilon=0)
			new_obs, rew, done, _ = eval_env.step(act)
			avg_rew += rew
			obs = new_obs
	avg_rew /= eval_episodes

	print('---------------------------------------')
	print(f'Avg. Evaluation over {eval_episodes} episodes: {avg_rew}')
	print('---------------------------------------')
	return avg_rew


def main():
	args = argparser()
	# Build environment
	env = gym.make(args.env_name)
	env.seed(args.seed)
	agent = DQN(args, env)
	obs = env.reset()

	# Episode related information
	ep_counter = 0
	ep_rew = 0

	# INITIAL LOGGING
	# Evaluate untrained policy
	logger = Logger('DQN', args.env_name, args.seed)
	avg_return = eval_policy(agent.policy, args.env_name, args.seed)
	logger.log_return(iteration=0, avg_return=avg_return)

	# Log hyperparameters and MLP details
	total_params = sum(p.numel() for p in agent.policy.parameters())
	trainable_params = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
	logger.log_params(hyperparameters=vars(args), total_params=total_params, trainable_params=trainable_params)

	for iteration in range(args.max_iterations):
		if args.render:
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
			print(f"episode: {ep_counter}, total reward: {ep_rew}, epsilon: {agent.epsilon}, iter: {iteration}")

			# Increment ep_counter after policy updates start
			ep_rew = 0
			ep_counter += 1

		# Update Policy
		loss = agent.update_policy()

		# LOG DATA
		# Evaluate policy
		if (iteration + 1) % args.eval_freq == 0:
			avg_return = eval_policy(agent.policy, args.env_name, args.seed)
			logger.log_return(iteration=iteration+1, avg_return=avg_return)

		# Log Model and Loss
		if iteration % args.save_freq == 0:
			logger.log_model(agent.policy)
			logger.log_state_dict(agent.optim.state_dict(), name='optim')
		if len(agent.memory) >= args.batch_size:
			logger.log_policy_update(iteration=iteration, loss=loss)

	# End
	env.close()
	p = Plotter()
	p.plot_individual('Avg. Return/Timestep', 'Timesteps', 'Return', 'DQN', args.env_name, args.seed, False)


if __name__ == '__main__':
	main()
