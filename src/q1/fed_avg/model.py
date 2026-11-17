'''
model.py

Contains the class for the neural 2 layer neural network for MNIST. Also has a module on pre processing the data.
'''

import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=62):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Hidden layer 1
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim), # Hidden layer 2
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

def pre_process(x):
    """Linearize the image, different handling for the Laplace perturbation."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 28 * 28)
    # assert x.min() >= 0.0 and x.max() <= 1.0, "unexpected pixel range" # Removed this so that we could perturb the data
    return 1.0 - x