import torch
from rlbotics.vpg.memory import Memory
from rlbotics.common.policies import MLPSoftmaxPolicy, MLPGaussianPolicy, MLPCritic

class VPG:
    def __init__(self, args, env):
        self.env = env

        # General parameters
        self.gamma = args.gamma
        self.lam = args.lam
        self.seed = args.seed
        self.memory_size = args.memory_size

        # Policy Network
        self.pi_lr = args.pi_lr
        self.pi_hidden_sizes = args.pi_hidden_sizes
        self.pi_activations = args.pi_activations
        self.pi_optimizer = args.pi_optimizer

        # Value Network
        self.v_lr = args.v_lr
        self.v_hidden_sizes = args.v_hidden_sizes
        self.v_activations = args.v_activations
        self.num_v_iters = args.num_v_iters
        self.v_optimizer = args.v_optimizer

        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape

        # Replay buffer
        self.memory = Memory(self.obs_dim, self.act_dim, self.memory_size, self.gamma, self.lam)

        self._build_policy()
        self._build_value_function()


    def _build_policy(self):
        continuous = False if len(self.env.action_space.shape) == 0 else True

        if continuous:
            self.obs_dim = self.env.observation_space.shape[0]
            self.act_dim = self.env.action_space.shape[0]

            self.policy = MLPGaussianPolicy(layer_sizes=[self.obs_dim] + self.pi_hidden_sizes + [self.act_dim],
                                           activations=self.pi_activations,
                                           seed=self.seed)
        else:
            self.obs_dim = self.env.observation_space.shape[0]
            self.act_dim = self.env.action_space.n

            self.policy = MLPSoftmaxPolicy(layer_sizes=[self.obs_dim] + self.pi_hidden_sizes + [self.act_dim],
                                           activations=self.pi_activations,
                                           seed=self.seed)

        # Set Optimizer
        if self.pi_optimizer == 'Adam':
            self.pi_optim = torch.optim.Adam(self.policy.parameters(), lr=self.pi_lr)
        elif self.pi_optimizer == 'RMSprop':
            self.pi_optim = torch.optim.RMSprop(self.policy.parameters(), lr=self.pi_lr)
        else:
            raise NameError(str(self.pi_optimizer) + ' Optimizer not supported')

    def _build_value_function(self):
        self.value = MLPCritic(layer_sizes=[self.obs_dim] + self.v_hidden_sizes,
                         activations=self.v_activations,
                         seed=self.seed)

        # Set Optimizer
        if self.v_optimizer == 'Adam':
            self.v_optim = torch.optim.Adam(self.value.parameters(), lr=self.v_lr)
        elif self.v_optimizer == 'RMSprop':
            self.v_optim = torch.optim.RMSprop(self.value.parameters(), lr=self.v_lr)
        else:
            raise NameError(str(self.v_optimizer) + ' Optimizer not supported')

    def store_transition(self, obs, act, rew):
        value = self.value(torch.as_tensor(obs, dtype=torch.float32)).detach().numpy()
        log_prob = self.policy._log_prob_from_distribution(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(act, dtype=torch.float32)).detach().numpy()

        self.memory.store(obs, act, rew, value, log_prob)

    def compute_policy_loss(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.policy(obs, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    def compute_value_loss(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.value(obs) - ret)**2).mean()

    def update_policy(self):
        self.data = self.memory.get()

        self.pi_optim.zero_grad()
        loss_pi, pi_info = self.compute_policy_loss(self.data)
        loss_pi.backward()
        self.pi_optim.step()
        return loss_pi.item()

    def update_value(self):
        for i in range(self.num_v_iters):
            self.v_optim.zero_grad()
            loss_v = self.compute_value_loss(self.data)
            loss_v.backward()
            self.v_optim.step()

        return loss_v.item()
