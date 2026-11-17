'''
run_parallel.py

Execution entry point.
'''

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import ray

import paths  # (must come before ray.init)
from parallel_server import ParallelServer

TRAIN_PATH = Path("data/Assignment3-data/train_data.npy")
TEST_PATH = Path("data/Assignment3-data/test_data.npy")

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
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # The defaults are the same hyperparams used for the sequential FedAvg.
    # ap.add_argument("--train", default=TRAIN_PATH)
    # ap.add_argument("--test", default=TEST_PATH)
    ap.add_argument("--rounds", type=int, default=300) # 300 Communication rounds just as in the sequential case.
    ap.add_argument("--epochs", type=int, default=10)  # using the same E from part 1, since that was optimal
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--clients-per-round", type=int, default=4)
    ap.add_argument("--gpus-per-client", type=float, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    res = ray.cluster_resources()
    print("Ray resources:", res)
    if res.get("CPU", 0) < args.clients_per_round:
        print(f"WARNING: only {res.get('CPU')} CPUs for {args.clients_per_round} actors, clients will not all run in parallel.")

    train_data = np.load(TRAIN_PATH, allow_pickle=True)
    test_data = np.load(TEST_PATH, allow_pickle=True)

    server = ParallelServer(train_data, test_data[0]["images"], test_data[0]["labels"], seed=args.seed)
    try:
        hist = server.train(args.rounds, args.epochs, args.lr, args.batch_size, m=args.clients_per_round, gpus_per_client=args.gpus_per_client)
    finally:
        ray.shutdown()

    for h in hist[::10]:
        print({k: round(v, 4) for k, v in h.items() if k != "clients"})
    print(f"mean round time: {np.mean([h['time'] for h in hist]):.2f}s")

    OUTPUT = Path("data/q1/parallel_clients")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PLOTS = Path("plots")
    PLOTS.mkdir(parents=True, exist_ok=True)
    
    plot_history(hist, f"Parallel FedAvg (m={args.clients_per_round}, E={args.epochs})", PLOTS / "parallel_fed_avg.png")

    test_loss, test_acc = server.evaluate_test() # once, at the end
    print(f"test loss={test_loss:.4f}, test acc={test_acc:.4f}")

    with open(OUTPUT / "parallel_history.json", "w") as f:
        json.dump({"args": vars(args), "test_loss": test_loss, "test_acc": test_acc, "history": hist}, f)