import torch
import torch.nn as nn

class PerturbationNetwork(nn.Module):
    """ Latent perturbation network """

    def __init__(self, latent_dim=100, num_hidden_layers=4, dropout=0.1):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = latent_dim*2

        perturb_list = []
        for i in range(num_hidden_layers):
            in_dim = self.latent_dim if i == 0 else self.hidden_size
            perturb_list += [
                nn.Linear(in_dim, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        perturb_list += [nn.Linear(self.hidden_size, self.latent_dim)]

        self._perturbation = nn.Sequential(*perturb_list)


    def forward(self, x):
        perturbation = self._perturbation(x)
        x = x + perturbation
        return x
