from collections import namedtuple


class Memory:
    def __init__(self):
        self.memory = []
        self.transition = namedtuple("transition", field_names=["obs", "act", "rew", "new_obs", "done", "log_prob", "val"])

    def add(self, obs, act, rew, new_obs, done, log_prob, val):
        transition = self.transition(obs=obs, act=act, rew=rew, new_obs=new_obs, done=done, log_prob=log_prob, val=val)
        self.memory.append(transition)

    def get_batch(self):
        transition_batch = self.transition(*zip(*self.memory))
        return transition_batch

    def __len__(self):
        return len(self.memory)

    def reset_memory(self):
        self.memory.clear()
