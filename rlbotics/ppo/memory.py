import numpy as np
import torch
from rlbotics.common.utils import combined_shape, discount_return

class Memory:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.step = 0
        self.trajectory_start_idx = 0
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        assert self.step < self.max_size     # buffer has to have room so you can store
        self.obs[self.step] = obs
        self.act[self.step] = act
        self.rew[self.step] = rew
        self.val[self.step] = val
        self.logp[self.step] = logp
        self.step += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.trajectory_start_idx, self.step)

        rews = np.append(self.rew[path_slice], last_val)
        vals = np.append(self.val[path_slice], last_val)

        # Calculate GAE advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[path_slice] = discount_return(deltas, self.gamma * self.lam)

        # Computes return
        self.ret[path_slice] = discount_return(rews, self.gamma)[:-1]

        self.trajectory_start_idx = self.step

    def get(self):
        assert self.step == self.max_size    # buffer has to be full before you can get
        self.step = 0
        self.trajectory_start_idx = 0
        data = dict(obs=self.obs, act=self.act, ret=self.ret,
                    adv=self.adv, logp=self.logp)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
