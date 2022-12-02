# https://spinningup.openai.com/en/latest/algorithms/ddpg.html
# General Parameters:
env_name       = 'LunarLanderContinuous-v2'
seed           = 0
gamma          = 0.99
max_iterations = 500000
eval_freq      = 1000		  # Has to divide max_iterations
render         = False
use_grad_clip  = False
save_freq      = 2000		  # Freq to save policy and q models

# DDPG Specific:
batch_size     = 128
buffer_size    = 1e6
polyak         = 0.005		  # Soft update for target network
act_noise      = 0.1		  # Stddev for Gaussian exploration noise added to policy at training time
noise_type     = 'gaussian'   # Gaussian or ou noise
random_steps   = 5e3		  # Random actions before training for exploration
update_after   = 1000		  # Number of env interactions to collect before training. Ensures replay buffer is full

# Policy Network Parameters
pi_lr           = 1e-3
pi_hidden_sizes = [400, 200] 	  # Dimensions have to be 1 less than activations
pi_activations  = ['relu', 'relu', 'tanh']
pi_optimizer    = 'Adam'

# Q Network Parameters
q_lr      	    = 1e-3
q_hidden_sizes  = [400, 200]	  # Dimensions have to be 1 less than activations
q_activations   = ['relu', 'relu', 'none']
q_optimizer     = 'Adam'
q_loss_type     = 'mse'
weight_decay    = 0				# L2 weight decay for the parameters

# Both Networks
weight_init     = None			# Initialise MLP params from (-x, x) either float or None
batch_norm      = False
