import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    Multi-Layered Perceptron
    """
    def __init__(self, layer_sizes, activations, seed, batch_norm=False, weight_init=None):
        """
        :param layer_sizes: (list) sizes of each layer (including IO layers)
        :param activations: (list)(strings) activations corresponding to each layer	e.g. ['relu', 'relu', 'none']
        :param weight_init: (None/float) uniform initialization for mlp params
        """
        super(MLP, self).__init__()
        torch.manual_seed(seed)
        self.obs_dim = layer_sizes[0]
        self.weight_init = weight_init

        # Build NN
        self.activations_dict = nn.ModuleDict({
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "softmax": nn.LogSoftmax(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(),
            "none": nn.Identity(),
        })

        # Build MLP and initialize weights if necessary
        self.mlp = self._build_mlp(layer_sizes, activations, batch_norm)
        if weight_init is not None:
            self.mlp.apply(self.init_weights)

    def _build_mlp(self, layer_sizes, activations, batch_norm):
        layers = []
        for i in range(len(layer_sizes)-1):
            if batch_norm:
                layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.BatchNorm1d(layer_sizes[i+1]),
                           self.activations_dict[activations[i]]]
            else:
                layers += [nn.Linear(layer_sizes[i], layer_sizes[i+1]), self.activations_dict[activations[i]]]
        return nn.Sequential(*layers)

    def init_weights(self, mlp):
        if type(mlp) == nn.Linear:
            nn.init.uniform_(mlp.weight, -self.weight_init, self.weight_init)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()

        x = x.view(-1, self.obs_dim)
        return self.mlp(x)

    def summary(self):
        print(self.mlp)
