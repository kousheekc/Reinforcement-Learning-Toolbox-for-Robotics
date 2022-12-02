# General Parameters:
env_name        = 'CartPole-v1'
gamma           = 0.99
lam             = 0.95
max_iterations  = 512
max_epochs      = 500
render          = False
num_v_iters     = 80
seed            = 0

# TRPO specific hyperparameters
kl_target =     0.003

# Policy Network:
pi_hidden_sizes     = [128, 128]
pi_activations      = ['relu', 'relu', 'none']
pi_lr               = 5e-4
pi_optimizer        = 'Adam'

# Value Network:
v_hidden_sizes      = [128, 128, 1]
v_activations       = ['relu', 'relu', 'relu', 'none']
v_lr                = 1e-3
v_optimizer        = 'Adam'
