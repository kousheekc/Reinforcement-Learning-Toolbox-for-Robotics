# General Parameters:
env_name        = 'LunarLanderContinuous-v2'
gamma           = 0.99
lam             = 0.95
max_iterations  = 1000
max_epochs      = 1000
render          = False
seed            = 9
memory_size     = 1000

# Policy Network:
pi_hidden_sizes     = [64, 64]
pi_activations      = ['tanh', 'tanh', 'none']
pi_lr               = 5e-4
pi_optimizer        = 'Adam'

# Value Network:
v_hidden_sizes      = [64, 64, 1]
v_activations       = ['tanh', 'tanh', 'none']
v_lr                = 1e-3
v_optimizer        = 'Adam'
num_v_iters        = 80
