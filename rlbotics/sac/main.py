import gym
import argparse

from rlbotics.sac.sac import SAC
import rlbotics.sac.hyperparameters as h
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
    parser.add_argument('--render', type=int, default=h.render)
    parser.add_argument('--alpha', type=float, default=h.alpha)
    parser.add_argument('--update_every', type=int, default=h.update_every)
    parser.add_argument('--save_freq', type=int, default=h.save_freq)

    # SAC Specific
    parser.add_argument('--batch_size', type=int, default=h.batch_size)
    parser.add_argument('--buffer_size', type=int, default=h.buffer_size)
    parser.add_argument('--polyak', type=float, default=h.polyak)
    parser.add_argument('--start_steps', type=int, default=h.start_steps)
    parser.add_argument('--update_after', type=int, default=h.update_after)

    # Policy and Q Network specific
    parser.add_argument('--pi_lr', type=float, default=h.pi_lr)
    parser.add_argument('--q_lr', type=float, default=h.q_lr)
    parser.add_argument('--pi_hidden_sizes', nargs='+', type=int, default=h.pi_hidden_sizes)
    parser.add_argument('--q_hidden_sizes', nargs='+', type=int, default=h.q_hidden_sizes)
    parser.add_argument('--pi_activations', nargs='+', type=str, default=h.pi_activations)
    parser.add_argument('--q_activations', nargs='+', type=str, default=h.q_activations)
    parser.add_argument('--pi_optimizer', type=str, default=h.pi_optimizer)
    parser.add_argument('--q_optimizer', type=str, default=h.q_optimizer)
    parser.add_argument('--q_loss_type', type=str, default=h.q_loss_type)

    return parser.parse_args()


def main():
    args = argparser()
    # Build environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    agent = SAC(args, env)
    obs = env.reset()

    # Episode related information (resume if necessary)
    ep_counter = 0
    ep_rew = 0

    for iteration in range(args.max_iterations):
        if args.render:
            env.render()

        # Take action Random in the beginning for exploration
        act = agent.get_action(obs) if iteration > args.start_steps else env.action_space.sample()
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
            if iteration > args.update_after:
                ep_counter += 1

        if iteration >= args.update_after and iteration % args.update_every == 0:
            for j in range(args.update_every):
                agent.update_actor_critic()

    # End
    env.close()
    p = Plotter()
    p.plot_individual('Episode/Reward', 'episodes', 'rewards', 'SAC', args.env_name, args.seed, True)


if __name__ == '__main__':
	main()
