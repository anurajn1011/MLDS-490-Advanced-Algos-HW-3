"""
non_standard_plot.py  -  the non standard plot shared by Part 1.1, Part 1.2 and Part 2.
Recommended to run using script: scripts/run_validation.sh plot

In every round each selected client returns an update delta_k = w_k - w_global.

We keep all of them and ask: do the clients push the global model in the same direction?
  * Top row (one heatmap per setting): x = cosine similarity between two clients' updates, y = communication round, colour = share of client pairs (pooled over a 5-round window). Mass at +1 means the clients agree, mass at 0 means their updates are unrelated.
  * Bottom: update coherence  |sum_k p_k delta_k| / sum_k p_k |delta_k|  per round (1 = all clients agree, 1/sqrt(m) = m unrelated updates of equal length, dotted).

It explains, with one picture, why larger E (client drift on non-IID data), fewer clients per round (Part 1.2, m=4) and larger Laplace noise (Part 2) make FedAvg noisier or slower.
The recorder reuses the exact seeds of Server.train, and validation_parallel.py shows that the parallel server is equivalent to the sequential one, so the m=4 setting is recorded with the sequential clients.

Run:  python non_standard_plot.py, QUICK=1 for a short version
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from val_common import add_paths, load_data, FIG, QUICK

add_paths("q1/fed_avg", "q2")
from clients import Client
from server import Server
from noisy_client import NoisyClient

def _vec(state):
    return torch.cat([v.detach().flatten().float() for v in state.values()]) # Detach unplugs from computational graph

def run_recording(server, clients, rounds, epochs, lr, batch_size, m=None):
    """
    Same loop and seeds as Server.train, but keeps every client's update. `clients` maps id to object with local_update (Client, NoisyClient, ...).
    """
    rec = {"cos": [], "coherence": [], "m": None, "val_acc": []}
    for r in range(rounds):
        ids = server.select_clients(m)
        g = {k: v.clone() for k, v in server.model.state_dict().items()}
        gv = _vec(g)
        states, deltas, ns, correct, tot = [], [], [], 0, 0
        for i in ids:
            sd, n, met = clients[i].local_update(g, epochs, lr, batch_size, seed=r * 1000 + i)
            states.append(sd); ns.append(n); deltas.append(_vec(sd) - gv)
            correct += met["val"][1]; tot += met["val"][2]
        server.aggregate(ids, states)
        D = torch.stack(deltas)
        w = torch.tensor(ns, dtype=torch.float32); w = w / w.sum()
        norms = D.norm(dim=1)
        coh = ((w[:, None] * D).sum(0).norm() / (w * norms).sum().clamp_min(1e-12)).item()
        Dn = D / norms[:, None].clamp_min(1e-12)
        iu = torch.triu_indices(len(ids), len(ids), 1)
        rec["cos"].append((Dn @ Dn.T)[iu[0], iu[1]].numpy())
        rec["coherence"].append(coh)
        rec["val_acc"].append(correct / max(tot, 1))
        rec["m"] = len(ids)
    return rec

def mean_cosine(rec, last=None):
    c = rec["cos"][-last:] if last else rec["cos"]
    return float(np.mean(np.concatenate(c)))

def plot_agreement(runs, path, window=5, bins=41):
    k = len(runs)
    fig = plt.figure(figsize=(4.3 * k, 8))
    gs = fig.add_gridspec(2, k, height_ratios=[3, 1.6])
    edges = np.linspace(-1, 1, bins + 1)
    for j, (name, rec) in enumerate(runs.items()):
        R = len(rec["cos"])
        H = np.zeros((R, bins))
        for r in range(R):
            pool = np.concatenate(rec["cos"][max(0, r - window + 1): r + 1])
            if len(pool):
                H[r] = np.histogram(pool, edges)[0] / len(pool)
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(H, aspect="auto", origin="lower", extent=[-1, 1, 0, R], cmap="magma")
        ax.axvline(0, color="w", lw=0.7, ls="--")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("cosine between two clients' updates")
        if j == 0:
            ax.set_ylabel("communication round")
        if j == k - 1:
            fig.colorbar(im, ax=ax, label="share of client pairs")
    ax = fig.add_subplot(gs[1, :])
    for name, rec in runs.items():
        c = np.array(rec["coherence"])
        sm = np.convolve(c, np.ones(window) / window, mode="valid")
        line, = ax.plot(np.arange(window - 1, len(c)), sm, label=name)
        ax.axhline(1 / np.sqrt(rec["m"]), color=line.get_color(), ls=":", lw=0.8)
    ax.set(xlabel="communication round", ylabel="update coherence", title="dotted: 1/sqrt(m), the value for m unrelated updates")
    ax.legend(fontsize=8)
    fig.suptitle("Do the selected clients push the global model the same way?")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__":
    train, test = load_data()
    N, R = len(train), (25 if QUICK else 300)
    E, LR, BATCH_SIZE, B = 10, 0.05, 32, 0.5

    def mk(clients, C=0.1):
        return Server(clients, test[0]["images"], test[0]["labels"], C=C, seed=0)

    clean = [Client(i, train[i]["images"], train[i]["labels"], seed=0) for i in range(N)]
    noisy = [NoisyClient(i, train[i]["images"], train[i]["labels"], noise_b=B, seed=0) for i in range(N)]
    lr_dp = LR * min(1.0, sum(c.clean_sq for c in noisy) / sum(c.noisy_sq for c in noisy))

    print("Constructing plots for each part...")
    runs = {
        f"Part 1.1: FedAvg, m=10, E={E}": run_recording(mk(clean), {c.id: c for c in clean}, R, E, LR, BATCH_SIZE),
        f"Part 1.2: parallel setting, m=4, E={E}": run_recording(mk(clean), {c.id: c for c in clean}, R, E, LR, BATCH_SIZE, m=4),
        f"Part 2: Laplace b={B}, m=10, E={E}, lr={lr_dp:.4f}": run_recording(mk(noisy), {c.id: c for c in noisy}, R, E, lr_dp, BATCH_SIZE),
    }
    for n, r in runs.items():
        print(f"{n}: mean cosine {mean_cosine(r):+.3f}, mean coherence {np.mean(r['coherence']):.3f}")
    plot_agreement(runs, FIG / "non_standard_plots.png")
    print("saved", FIG / "non_standard_plots.png")