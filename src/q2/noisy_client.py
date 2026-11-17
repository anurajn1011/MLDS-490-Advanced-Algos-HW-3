'''
noisy_client.py

Class for adding noise to the data, according to Laplace distribution.
'''

import numpy as np
import torch

import paths
from clients import Client


def perturb(x, b, rng, clip=False):
    """X' = X + eps, eps_ij distributed according to Laplace(0, b), independent per pixel."""
    eps = rng.laplace(loc=0.0, scale=b, size=tuple(x.shape)).astype(np.float32)
    out = x + torch.from_numpy(eps)
    return out.clamp(0.0, 1.0) if clip else out

class NoisyClient(Client):
    """Same split and training as Client; the local data is perturbed once at construction."""

    def __init__(self, client_id, X, y, noise_b=0.0, val_frac=0.2, seed=0, clip=False):
        super().__init__(client_id, X, y, val_frac=val_frac, seed=seed)  # clean split + scaling, also pre processes so that the image is on a dark background. 
        self.noise_b = noise_b
        self.clean_sq = self.X_train.pow(2).sum(1).mean().item()   # mean ||x||^2, for lr scaling

        if noise_b > 0:
            # separate RNG stream from the split seed; fixed once, never resampled
            rng = np.random.default_rng([seed, 10_000, client_id])
            self.X_train = perturb(self.X_train, noise_b, rng, clip)
            self.X_val = perturb(self.X_val, noise_b, rng, clip)

        self.noisy_sq = self.X_train.pow(2).sum(1).mean().item()