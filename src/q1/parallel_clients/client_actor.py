"""
client_actor.py

Wrapper for existing client code for Ray.
"""

import ray
import torch

import paths 
from clients import Client

@ray.remote(num_cpus=1, num_gpus=0) # at most 1 CPU, 0 GPUs by default
class ClientActor:
    def __init__(self, client_id, X, y, val_frac=0.2, seed=0):
        torch.set_num_threads(1) # one actor = one CPU; avoid oversubscription
        self.client = Client(client_id, X, y, val_frac=val_frac, seed=seed)

    def local_update(self, global_state, epochs, lr, batch_size, seed=0):
        return self.client.local_update(global_state, epochs, lr, batch_size, seed=seed)