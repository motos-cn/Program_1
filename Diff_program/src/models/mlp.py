import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_features, hidden_dims=None):
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Identity())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)