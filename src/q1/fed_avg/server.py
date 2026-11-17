"""
server.py

The server class, serving as the global model.
"""

import numpy as np
import torch
import torch.nn as nn
from model import MLP, pre_process

class Server:
    def __init__(self, clients, X_test, y_test, C=0.1, seed=0):
        self.clients = {c.id: c for c in clients}
        self.n_train = {c.id: c.n_train for c in clients}  # server knows sample counts
        self.C = C
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed) # reproducible model init
        self.model = MLP()
        self.X_test = torch.from_numpy(pre_process(X_test))
        self.y_test = torch.as_tensor(y_test, dtype=torch.long)

    def select_clients(self, m=None):
        ids = list(self.clients)
        m = m if m is not None else max(1, int(self.C * len(ids)))
        assert 1 <= m <= len(ids), f"invalid number of clients per round: {m}"
        return self.rng.choice(ids, size=m, replace=False).tolist()

    def aggregate(self, ids, states):
        # weighted average using the server's own sample count table
        assert len(states) > 0, "no client updates to aggregate"
        total = sum(self.n_train[i] for i in ids)
        new_state = {
            k: sum(sd[k] * (self.n_train[i] / total) for i, sd in zip(ids, states))
            for k in states[0]
        }
        self.model.load_state_dict(new_state)

    @torch.no_grad()
    def evaluate_test(self):
        self.model.eval()
        logits = self.model(self.X_test)
        loss = nn.functional.cross_entropy(logits, self.y_test).item()
        acc = (logits.argmax(1) == self.y_test).float().mean().item()
        return loss, acc

    def train(self, rounds, epochs, lr, batch_size, m=None):
        history = []
        for r in range(rounds):
            ids = self.select_clients(m)
            global_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            states, tr, va = [], [0.0, 0, 0], [0.0, 0, 0]
            for i in ids:
                sd, n, met = self.clients[i].local_update(global_state, epochs, lr, batch_size, seed=r * 1000 + i) # Shuffles batches differently every time
                assert n == self.n_train[i]
                states.append(sd)
                for acc_, key in ((tr, "train"), (va, "val")):
                    for j in range(3):
                        acc_[j] += met[key][j]
            self.aggregate(ids, states)
            history.append({
                "round": r, "clients": ids,
                "train_loss": tr[0] / tr[2], "train_acc": tr[1] / tr[2],
                "val_loss": va[0] / va[2],   "val_acc": va[1] / va[2],
            })
        return history