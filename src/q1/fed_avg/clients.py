"""
clients.py

Class method for client training and communication.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import MLP, pre_process

class Client:
    def __init__(self, client_id, X, y, val_frac=0.2, seed=0):
        self.id = client_id
        X = torch.from_numpy(pre_process(X))
        y = torch.as_tensor(y, dtype=torch.long)

        g = torch.Generator().manual_seed(seed + client_id)
        perm = torch.randperm(len(X), generator=g) # Random shuffling of the indices
        n_val = int(val_frac * len(X)) 
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        self.X_train, self.y_train = X[train_idx], y[train_idx]
        self.X_val, self.y_val = X[val_idx], y[val_idx]
        self.n_train, self.n_val = len(self.X_train), len(self.X_val)

    @torch.no_grad()
    def _evaluate(self, model, X, y):
        model.eval()
        logits = model(X)
        loss_sum = nn.functional.cross_entropy(logits, y, reduction="sum").item() # We use sum to avoid errors in batch size
        correct = (logits.argmax(1) == y).sum().item()
        return loss_sum, correct, len(y)

    def local_update(self, global_state, epochs, lr, batch_size, seed=0):
        model = MLP()
        model.load_state_dict(global_state)
        opt = torch.optim.SGD(model.parameters(), lr=lr) # SGD
        loss_fn = nn.CrossEntropyLoss()

        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=batch_size, shuffle=True, generator=g)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

        metrics = {
            "train": self._evaluate(model, self.X_train, self.y_train),
            "val": self._evaluate(model, self.X_val, self.y_val),
        }
        new_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return new_state, self.n_train, metrics