import gym
import argparse
import numpy as np

from rlbotics.ddpg.ddpg import DDPG
from rlbotics.common.logger import Logger
import rlbotics.ddpg.hyperparameters as h
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
	parser.add_argument('--max_iterations', type=int, default=h.max_iterations)
	parser.add_argument('--eval_freq', type=int, default=h.eval_freq)
	parser.add_argument('--render', type=int, default=h.render)
	parser.add_argument('--use_grad_clip', type=int, default=h.use_grad_clip)

	# DDPG Specific
	parser.add_argument('--batch_size', type=int, default=h.batch_size)
	parser.add_argument('--buffer_size', type=int, default=h.buffer_size)
	parser.add_argument('--polyak', type=float, default=h.polyak)
	parser.add_argument('--act_noise', type=float, default=h.act_noise)
	parser.add_argument('--noise_type', type=str, default=h.noise_type)
	parser.add_argument('--random_steps', type=int, default=h.random_steps)
	parser.add_argument('--update_after', type=int, default=h.update_after)

	# Policy and Q Network specific
	parser.add_argument('--save_freq', type=int, default=h.save_freq)
	parser.add_argument('--pi_lr', type=float, default=h.pi_lr)
	parser.add_argument('--q_lr', type=float, default=h.q_lr)
	parser.add_argument('--pi_hidden_sizes', nargs='+', type=int, default=h.pi_hidden_sizes)
	parser.add_argument('--q_hidden_sizes', nargs='+', type=int, default=h.q_hidden_sizes)
	parser.add_argument('--pi_activations', nargs='+', type=str, default=h.pi_activations)
	parser.add_argument('--q_activations', nargs='+', type=str, default=h.q_activations)
	parser.add_argument('--pi_optimizer', type=str, default=h.pi_optimizer)
	parser.add_argument('--q_optimizer', type=str, default=h.q_optimizer)
	parser.add_argument('--q_loss_type', type=str, default=h.q_loss_type)
	parser.add_argument('--weight_decay', type=float, default=h.weight_decay)
	parser.add_argument('--weight_init', type=float, default=h.weight_init)
	parser.add_argument('--batch_norm', type=int, default=h.batch_norm)

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
			policy.eval()
			act = policy.get_action(obs).detach().numpy()
			act = np.clip(act, -policy.act_lim, policy.act_lim)[0]
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
	agent = DDPG(args, env)
	obs = env.reset()

	# Episode related information
	ep_counter = 0
	ep_rew = 0

	# INITIAL LOGGING
	# Evaluate untrained policy
	logger = Logger('DDPG', args.env_name, args.seed)
	avg_return = eval_policy(agent.pi, args.env_name, args.seed)
	logger.log_return(iteration=0, avg_return=avg_return)

	# Log hyperparameters and MLP details
	total_params = sum(p.numel() for p in agent.pi.parameters()) + sum(p.numel() for p in agent.q.parameters())
	trainable_params = sum(p.numel() for p in agent.pi.parameters() if p.requires_grad)\
					   + sum(p.numel() for p in agent.q.parameters() if p.requires_grad)
	logger.log_params(hyperparameters=vars(args), total_params=total_params, trainable_params=trainable_params)

	for iteration in range(args.max_iterations):
		if args.render:
			env.render()

		# Take action Random in the beginning for exploration
		act = agent.get_action(obs) if iteration > args.random_steps else env.action_space.sample()
		new_obs, rew, done, _ = env.step(act)

		# Store experience
		agent.store_transition(obs, act, rew, new_obs, done)
		ep_rew += rew
		obs = new_obs

		# Episode done
		if done:
			obs = env.reset()

			# Display results
			print(f"episode: {ep_counter}, total reward: {ep_rew}, timesteps: {iteration}")

			# Increment ep_counter after policy updates start
			ep_rew = 0
			ep_counter += 1

		# Update Actor Critic
		if iteration >= args.random_steps:
			q_loss, pi_loss = agent.update_actor_critic()

		# LOG DATA
		# Evaluate policy
		if (iteration + 1) % args.eval_freq == 0:
			avg_return = eval_policy(agent.pi, args.env_name, args.seed)
			logger.log_return(iteration=iteration+1, avg_return=avg_return)

		# Log Model and Loss
		if iteration % args.save_freq == 0 or iteration+1 == args.max_iterations:
			logger.log_model(agent.q, name='q')
			logger.log_model(agent.pi, name='pi')
			logger.log_model(agent.q_target, name='q_targ')
			logger.log_model(agent.pi_target, name='pi_targ')
			logger.log_state_dict(agent.q_optim.state_dict(), name='q_optim')
			logger.log_state_dict(agent.pi_optim.state_dict(), name='pi_optim')
		if iteration >= args.random_steps:
			logger.log_policy_update(iteration=iteration, q_loss=q_loss, pi_loss=pi_loss)

	# End
	env.close()
	p = Plotter()
	p.plot_individual('Avg. Return/Timestep', 'Timesteps', 'Return', 'DDPG', args.env_name, args.seed, False)


if __name__ == '__main__':
	main()
