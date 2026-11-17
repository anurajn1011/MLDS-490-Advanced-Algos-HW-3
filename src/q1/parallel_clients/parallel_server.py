'''
parallel_server.py

Subclass of server class, updated training method.
'''

import time
from collections import namedtuple
import ray
import paths
from server import Server
from client_actor import ClientActor

ClientInfo = namedtuple("ClientInfo", ["id", "n_train"])

class ParallelServer(Server):
    def __init__(self, train_data, X_test, y_test, C=0.1, seed=0, val_frac=0.2):
        # n_train must match Client's split: n - int(val_frac * n)
        infos = []
        for i in range(len(train_data)):
            n = len(train_data[i]["labels"])
            infos.append(ClientInfo(i, n - int(val_frac * n)))
        super().__init__(infos, X_test, y_test, C=C, seed=seed)
        self.train_data = train_data
        self.val_frac = val_frac
        self.client_seed = seed

    def train(self, rounds, epochs, lr, batch_size, m=4, gpus_per_client=0):
        history = []
        for r in range(rounds):
            t0 = time.perf_counter()
            ids = self.select_clients(m)

            # put the global weights in the object store once, share with all actors
            state_ref = ray.put({k: v.clone() for k, v in self.model.state_dict().items()})

            actors = []
            try:
                # actors exist only for the clients selected this round
                actors = [
                    ClientActor.options(num_cpus=1, num_gpus=gpus_per_client).remote(
                        i, self.train_data[i]["images"], self.train_data[i]["labels"],
                        self.val_frac, self.client_seed)
                    for i in ids
                ]
                futures = [
                    a.local_update.remote(state_ref, epochs, lr, batch_size, seed=r * 1000 + i)
                    for a, i in zip(actors, ids)
                ]
                results = ray.get(futures) # runs in parallel, order matches ids
            finally:
                for a in actors:
                    ray.kill(a) # free actor memory before the next round

            states, tr, va = [], [0.0, 0, 0], [0.0, 0, 0]
            for i, (sd, n, met) in zip(ids, results):
                assert n == self.n_train[i]
                states.append(sd)
                for acc_, key in ((tr, "train"), (va, "val")):
                    for j in range(3):
                        acc_[j] += met[key][j]
            self.aggregate(ids, states)

            history.append({
                "round": r, "clients": ids, "time": time.perf_counter() - t0,
                "train_loss": tr[0] / tr[2], "train_acc": tr[1] / tr[2],
                "val_loss": va[0] / va[2],   "val_acc": va[1] / va[2],
            })
            h = history[-1]
            if r % 5 == 0 or r == rounds - 1:
                print(f"[round {r+1}/{rounds}] clients={ids} train_acc={h['train_acc']:.3f} val_acc={h['val_acc']:.3f} val_loss={h['val_loss']:.3f} time={h['time']:.1f}s", flush=True)
        return history