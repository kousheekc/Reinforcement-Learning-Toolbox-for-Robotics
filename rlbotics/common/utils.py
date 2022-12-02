import torch
import pandas as pd
from statistics import mean
import numpy as np
import scipy.signal


# TODO: Cleanup unused functions
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_return(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# def get_latest_ep_return(log_file):
#     log = pd.read_csv(log_file)
#     l = len(log)
#     ep_return = log.loc[l-1, 'rewards']
#
#     for i in reversed(range(l-1)):
#         if log.loc[i,'done'] == False:
#             ep_return += log.loc[i, 'rewards']
#
#         elif log.loc[i,'done'] == True:
#             break
#
#     return ep_return


# def get_ep_returns(log_file, epoch_iter=1):
#     """
#     :param log_file: file where transitions.csv is found
#     :param epoch_iter: number of iterations for one epoch. (hyperparameters:max_iterations)
#     :return: (list) of returns from each episode
#     """
#     logs = pd.read_csv(log_file)
#
#     ep_sum = 0
#     temp = []
#     ep_returns = []
#
#     for i in range(len(logs)):
#         if logs.loc[i,'done'] == False:
#             ep_sum += logs.loc[i, 'rewards']
#
#         elif logs.loc[i,'done'] == True:
#             ep_sum += logs.loc[i, 'rewards']
#             temp.append(ep_sum)
#             ep_sum = 0
#
#         if i % epoch_iter == 0 and len(temp) != 0:
#             ep_returns.append(mean(temp))
#             temp.clear()
#
#     return ep_returns


def get_expected_return(rew, done, gamma, normalize_output=True):
    g = torch.zeros_like(rew, dtype=torch.float32)
    cumulative = 0.0
    for k in reversed(range(len(rew))):
        cumulative = rew[k] + gamma * cumulative * (1.0 - done[k])
        g[k] = cumulative
    if normalize_output:
        normalize(g)
    return g


def GAE(rew, done, val, gamma, lam, normalize_output=True):
    rew = torch.cat((rew, torch.tensor([0], dtype=torch.float32)))
    val = torch.cat((val, torch.tensor([0], dtype=torch.float32)))

    td_errors = rew[:-1] + gamma * val[1:] * (1 - done) - val[:-1]

    adv = get_expected_return(td_errors, done, gamma*lam, normalize_output)

    return adv


def normalize(x):
    return (x - torch.mean(x)) / torch.std(x)
