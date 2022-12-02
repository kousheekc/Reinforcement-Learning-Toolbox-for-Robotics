import torch
import itertools
import numpy as np
from copy import deepcopy

from rlbotics.common.loss import losses
from rlbotics.common.logger import Logger
from rlbotics.sac.replay_buffer import ReplayBuffer
from rlbotics.common.policies import MLPSquashedGaussianPolicy, MLPQFunctionContinuous

class SAC:
    """
    NOTE: For Continuous environments only
    """
    def __init__(self, args, env):
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.act_lim = env.action_space.high[0]

        # General parameters
        self.seed = args.seed
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.save_freq = args.save_freq

        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.polyak = args.polyak
        self.start_steps = args.start_steps
        self.update_after = args.update_after

        self.pi_lr = args.pi_lr
        self.pi_hidden_sizes = args.pi_hidden_sizes
        self.pi_activations = args.pi_activations
        self.pi_optimizer = args.pi_optimizer

        self.q_lr = args.q_lr
        self.q_hidden_sizes = args.q_hidden_sizes
        self.q_activations = args.q_activations
        self.q_optimizer = args.q_optimizer
        self.q_loss_type = args.q_loss_type

        # Set device
        gpu = 0
        self.device = torch.device(f"cuda:{gpu}"if torch.cuda.is_available() else "cpu")

        # Replay buffer
        self.memory = ReplayBuffer(self.buffer_size, self.seed)

        # Logger
        self.logger = Logger('SAC', args.env_name, self.seed)

        # Steps
        self.steps_done = 0

        # Loss function
        self.q_criterion = losses(self.q_loss_type)

        # Build pi and q Networks
        # Resume training if necessary
        self._build_policy()
        self._build_q_function()

        # Log parameter data
        total_params = sum(p.numel() for p in self.pi.parameters())
        trainable_params = sum(p.numel() for p in self.pi.parameters() if p.requires_grad)
        self.logger.log(hyperparameters=vars(args), total_params=total_params, trainable_params=trainable_params)

    def _build_policy(self):
        layer_sizes = [self.obs_dim] + self.pi_hidden_sizes + [self.act_dim]
        self.pi = MLPSquashedGaussianPolicy(act_lim=self.act_lim,
                                           layer_sizes=layer_sizes,
                                           activations=self.pi_activations,
                                           seed=self.seed)

        self.pi.summary()

        # Set Optimizer
        if self.pi_optimizer == 'Adam':
            self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=self.pi_lr)
        elif self.pi_optimizer == 'RMSprop':
            self.pi_optim = torch.optim.RMSprop(self.pi.parameters(), lr=self.pi_lr)
        else:
            raise NameError(str(self.pi_optimizer) + ' Optimizer not supported')

        # Build Target
        self.pi_target = deepcopy(self.pi)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.pi_target.parameters():
            p.requires_grad = False

    def _build_q_function(self):
        layer_sizes = [self.obs_dim + self.act_dim] + self.q_hidden_sizes + [1]
        self.q1 = MLPQFunctionContinuous(layer_sizes=layer_sizes,
                                        activations=self.q_activations,
                                        seed=self.seed)

        self.q2 = MLPQFunctionContinuous(layer_sizes=layer_sizes,
                                        activations=self.q_activations,
                                        seed=self.seed)

        self.q1.summary()
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())

        # Set Optimizer
        if self.q_optimizer == 'Adam':
            self.q_optim = torch.optim.Adam(self.q_params, lr=self.q_lr)
        elif self.q_optimizer == 'RMSprop':
            self.q_optim = torch.optim.RMSprop(self.q_params, lr=self.q_lr)
        else:
            raise NameError(str(self.q_optimizer) + ' Optimizer not supported')

        # Build Target
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.q_targ_params = itertools.chain(self.q1_target.parameters(), self.q2_target.parameters())

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p1, p2 in zip(self.q1_target.parameters(), self.q2_target.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

    def store_transition(self, obs, act, rew, new_obs, done):
        self.memory.add(obs, act, rew, new_obs, done)
        self.steps_done += 1

        # Log Done, reward data
        if self.steps_done > self.update_after:
            self.logger.log(name='transitions', done=done, rewards=rew)

    def get_action(self, obs):
        self.pi.eval()
        action = self.pi.get_action(obs)[0].detach().numpy()
        return np.clip(action, -self.act_lim, self.act_lim)[0]

    def _compute_q_loss(self, batch):
        obs_batch = torch.as_tensor(batch.obs, dtype=torch.float)
        act_batch = torch.as_tensor(batch.act)
        rew_batch = torch.as_tensor(batch.rew).unsqueeze(1)
        new_obs_batch = torch.as_tensor(batch.new_obs, dtype=torch.float)
        done_batch = torch.as_tensor(batch.done)
        not_done_batch = torch.logical_not(done_batch).unsqueeze(1)

        q1_pred = self.q1.get_q_value(obs_batch, act_batch)
        q2_pred = self.q2.get_q_value(obs_batch, act_batch)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            pi_targ, logp_pi = self.pi.get_action(new_obs_batch)

            # Target Q-values
            q1_pi_targ = self.q1_target.get_q_value(new_obs_batch, pi_targ)
            q2_pi_targ = self.q2_target.get_q_value(new_obs_batch, pi_targ)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = rew_batch + self.gamma * not_done_batch * (q_pi_targ - self.alpha * logp_pi)

        q1_loss = self.q_criterion(q1_pred, backup.float())
        q2_loss = self.q_criterion(q2_pred, backup.float())
        loss = q1_loss + q2_loss

        return loss

    def _compute_pi_loss(self, batch):
        obs_batch = torch.as_tensor(batch.obs, dtype=torch.float)

        pi, logp_pi = self.pi.get_action(obs_batch)
        q1_pi = self.q1.get_q_value(obs_batch, pi)
        q2_pi = self.q2.get_q_value(obs_batch, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update_actor_critic(self):
        if self.steps_done < self.update_after or len(self.memory) < self.batch_size:
            return

        # Sample batch of transitions
        transition_batch = self.memory.sample(self.batch_size)

        q_loss = self._compute_q_loss(transition_batch)

        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()

        for p in self.q_params:
            p.requires_grad = False

        pi_loss, pi_info = self._compute_pi_loss(transition_batch)

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        # Update Target Networks
        self._update_target_actor_critic()

        # Log pi loss
        self.logger.log(name='policy_updates', pi_loss=pi_loss.item())
        self.logger.log(name='policy_updates', q_loss=q_loss.item())

        # Log Model/Optimizer, Loss and # iterations and episodes
        if self.steps_done % self.save_freq == 0:
            self.logger.log_model(self.q1, name='q1')
            self.logger.log_model(self.q2, name='q2')
            self.logger.log_model(self.pi, name='pi')
            self.logger.log_model(self.q1_target, name='q1_targ')
            self.logger.log_model(self.q2_target, name='q2_targ')
            self.logger.log_model(self.pi_target, name='pi_targ')
            self.logger.log_state_dict(self.q_optim.state_dict(), name='q_optim')
            self.logger.log_state_dict(self.pi_optim.state_dict(), name='pi_optim')

    def _update_target_actor_critic(self):
        # Polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.q_params, self.q_targ_params):
                p_targ.data.copy_(self.polyak*p.data + (1-self.polyak)*p_targ.data)
            for p, p_targ in zip(self.pi.parameters(), self.pi_target.parameters()):
                p_targ.data.copy_(self.polyak*p.data + (1-self.polyak)*p_targ.data)
