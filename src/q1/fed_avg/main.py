'''
main.py

Implementation of the Federated Averaging algorithm. Objective is to train the MNIST dataset. 

Run this script from the base of the directory:
    python3 src/q1/fed_avg/main.py

There are 100 clients. 
'''

import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from clients import Client
from server import Server

TRAIN_PATH = Path("data/Assignment3-data/train_data.npy")
TEST_PATH = Path("data/Assignment3-data/test_data.npy")
SEED = 0

def run(C=0.1, epochs=1, lr=0.05, batch_size=32, rounds=100, m=None, seed=SEED):
    train_data = np.load(TRAIN_PATH, allow_pickle=True)
    test_data = np.load(TEST_PATH, allow_pickle=True)

    clients = [
        Client(i, train_data[i]["images"], train_data[i]["labels"], seed=seed)
        for i in range(len(train_data))
    ]
    server = Server(clients, test_data[0]["images"], test_data[0]["labels"],
                    C=C, seed=seed)
    hist = server.train(rounds, epochs, lr, batch_size, m=m)
    return server, hist

def plot_history(hist, title, path):
    r = [h["round"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("train_loss", "val_loss"):
        ax[0].plot(r, [h[k] for h in hist], label=k)
    for k in ("train_acc", "val_acc"):
        ax[1].plot(r, [h[k] for h in hist], label=k)
    ax[0].set(xlabel="Communication round", ylabel="Loss", title=title)
    ax[1].set(xlabel="Communication round", ylabel="Accuracy", title=title)
    ax[0].legend(); ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def sweep(Cs, Es, lr=0.05, batch_size=32, rounds=300, seed=SEED):
    results = {}
    for C, E in itertools.product(Cs, Es):
        print(f"running C={C}, E={E}")
        _, hist = run(C=C, epochs=E, lr=lr, batch_size=batch_size,
                      rounds=rounds, seed=seed)
        results[(C, E)] = hist
    return results

def summarize(results, last=10):
    # mean of the last `last` rounds smooths out client sampling noise
    print(f"{'C':>6} {'E':>3} {'train_acc':>10} {'val_acc':>8}")
    for (C, E), h in results.items():
        tr = np.mean([x["train_acc"] for x in h[-last:]])
        va = np.mean([x["val_acc"] for x in h[-last:]])
        print(f"{C:>6} {E:>3} {tr:>10.3f} {va:>8.3f}")

def plot_sweep(results, Cs, Es, path="sweep.png"):
    fig, axes = plt.subplots(2, len(Cs), figsize=(5 * len(Cs), 8), sharex=True, sharey="row")
    axes = np.atleast_2d(axes).reshape(2, len(Cs))
    for j, C in enumerate(Cs):
        for E in Es:
            h = results[(C, E)]
            r = [x["round"] for x in h]
            axes[0, j].plot(r, [x["train_acc"] for x in h], label=f"E={E}")
            axes[1, j].plot(r, [x["val_acc"] for x in h], label=f"E={E}")
        axes[0, j].set_title(f"C={C}: training accuracy")
        axes[1, j].set_title(f"C={C}: validation accuracy")
        axes[1, j].set_xlabel("Communication round")
    axes[0, 0].set_ylabel("Accuracy"); axes[1, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Cs, Es = [0.02, 0.05, 0.10], [1, 5, 10]
    # results = sweep(Cs, Es)
    # summarize(results)
    OUTPUT = Path("data/q1/fed_avg")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PLOTS = Path("plots")
    PLOTS.mkdir(parents=True, exist_ok=True)
    # plot_sweep(results, Cs, Es, PLOTS / "fed_avg_sweep.png")
    # with open(OUTPUT / "sweep_history.json", "w") as f:
    #     json.dump({f"C={c}_E={e}": h for (c, e), h in results.items()}, f)

    # From the above resulting table and plot, we find that E = 10 and C = 0.1 performs best. 
    BEST_C, BEST_E = 0.10, 10
    server, hist = run(C=BEST_C, epochs=BEST_E, lr=0.05, batch_size=32, rounds=300)
    plot_history(hist, f"FedAvg C={BEST_C}, E={BEST_E}", OUTPUT / "fed_avg_final.png")
    test_loss, test_acc = server.evaluate_test()
    print(f"test loss={test_loss:.4f}, test acc={test_acc:.4f}")