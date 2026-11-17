'''
run_dp.py

Run the differential policy model and generate required data and plots.
'''

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

import paths
from noisy_client import NoisyClient, perturb
from server import Server
from model import pre_process

TRAIN_PATH = Path("data/Assignment3-data/train_data.npy")
TEST_PATH = Path("data/Assignment3-data/test_data.npy")

def run(train_data, test_data, noise_b, C=0.1, epochs=10, lr=0.05, batch_size=32, rounds=300, seed=0, clip=False, scale_lr=False):
    clients = [NoisyClient(i, train_data[i]["images"], train_data[i]["labels"], noise_b=noise_b, seed=seed, clip=clip) for i in range(len(train_data))]
    lr_eff = lr
    if scale_lr and noise_b > 0: # keep lr * ||x||^2 roughly constant, we want to scale with noise
        ratio = sum(c.clean_sq for c in clients) / sum(c.noisy_sq for c in clients)
        lr_eff = lr * min(1.0, ratio)
    server = Server(clients, test_data[0]["images"], test_data[0]["labels"], C=C, seed=seed)
    hist = server.train(rounds, epochs, lr_eff, batch_size)
    test_loss, test_acc = server.evaluate_test()   # test data is clean and untouched by noise
    return hist, test_loss, test_acc, lr_eff

def final(hist, key, last=10):
    return float(np.mean([h[key] for h in hist[-last:]]))

def plot_curves(hist, title, path):
    r = [h["round"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("train_acc", "val_acc"):
        ax[0].plot(r, [h[k] for h in hist], label=k)
    for k in ("train_loss", "val_loss"):
        ax[1].plot(r, [h[k] for h in hist], label=k)
    ax[0].set(xlabel="Communication round", ylabel="Accuracy", title=title)
    ax[1].set(xlabel="Communication round", ylabel="Loss", title=title)
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

def sweep(train_data, test_data, bs, seeds, **kw):
    out = {}
    for b in bs:
        out[b] = []
        for s in seeds:
            hist, tl, ta, lr_eff = run(train_data, test_data, b, seed=s, **kw)
            row = {"seed": s, "lr": lr_eff, "train_acc": final(hist, "train_acc"),
                   "val_acc": final(hist, "val_acc"), "test_acc": ta, "test_loss": tl,
                   "history": hist}
            out[b].append(row)
            print(f"b={b:<5} seed={s} lr={lr_eff:.4f} train={row['train_acc']:.3f} "
                  f"val={row['val_acc']:.3f} test={ta:.3f}")
    return out

def plot_sweep(results, path):
    bs = sorted(results)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in (("train_acc", "Final training accuracy"),
                       ("test_acc", "Test accuracy")):
        m = np.array([np.mean([r[key] for r in results[b]]) for b in bs])
        s = np.array([np.std([r[key] for r in results[b]]) for b in bs])
        ax.errorbar(bs, m, yerr=s, marker="o", capsize=3, label=label)
    ax.set_xscale("symlog", linthresh=0.05)  # b=0 (no noise) stays visible
    ax.set(xlabel="Laplace noise scale b", ylabel="Accuracy",
           title="FedAvg accuracy vs. input noise scale")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_examples(train_data, bs, path, seed=0):
    x =torch.from_numpy(pre_process(train_data[0]["images"][:4]))
    fig, axes = plt.subplots(len(bs), 4, figsize=(8, 2 * len(bs)))
    for i, b in enumerate(bs):
        rng = np.random.default_rng(seed)
        xn = perturb(x, b, rng) if b > 0 else x
        for j in range(4):
            axes[i, j].imshow(xn[j].reshape(28, 28), cmap="gray_r")
            axes[i, j].axis("off")
        axes[i, 0].set_title(f"b={b}", fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--train", default="train_data.npy")
    # ap.add_argument("--test", default="test_data.npy")
    ap.add_argument("--b", type=float, default=0.5, help="noise scale for a single run")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bs", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1, 2, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=10) # Chosen E from part 1
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=300) # To remain consistent with the other parts
    ap.add_argument("--clip", action="store_true", help="clip perturbed pixels to [0,1]") # Do not use
    ap.add_argument("--scale-lr", action="store_true", help="shrink lr as input norm grows") # Do not use
    ap.add_argument("--show-examples", action="store_true")
    # ap.add_argument("--out", default="results")
    args = ap.parse_args()
    # os.makedirs(args.out, exist_ok=True)

    OUTPUT = Path("data/q2")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PLOTS = Path("plots")
    PLOTS.mkdir(parents=True, exist_ok=True)
    
    train_data = np.load(TRAIN_PATH, allow_pickle=True)
    test_data = np.load(TEST_PATH, allow_pickle=True)
    kw = dict(C=args.C, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, rounds=args.rounds, clip=args.clip, scale_lr=args.scale_lr)

    if args.show_examples:
        print(f"Plotting noise examples to {PLOTS} / dp_noise_examples.png")
        plot_examples(train_data, args.bs, PLOTS / "dp_noise_examples.png")

    if args.sweep:
        print(f"Sweeping over different b values, saving plot to {PLOTS} / dp_noise_examples.png and saving outputs to {OUTPUT} / dp_sweep.json")
        results = sweep(train_data, test_data, args.bs, args.seeds, **kw)
        plot_sweep(results, PLOTS / "dp_accuracy_vs_noise.png")
        with open(OUTPUT / "dp_sweep.json", "w") as f:
            json.dump({"args": vars(args), "results": {str(b): r for b, r in results.items()}}, f)
    else:
        print("+"*10, f"RUNNING A DIFFERENTIAL PRIVACY RUN with b = {args.b}", "+"*10)
        hist, tl, ta, lr_eff = run(train_data, test_data, args.b, seed=args.seeds[0], **kw)
        plot_curves(hist, f"FedAvg with Laplace noise b={args.b}", PLOTS / f"dp_curves_b{args.b}.png")
        print(f"b={args.b} lr={lr_eff:.4f} final train acc={final(hist, 'train_acc'):.3f} val acc={final(hist, 'val_acc'):.3f} | test loss={tl:.4f}, test acc={ta:.4f}")
        with open(OUTPUT / f"dp_run_b{args.b}.json", "w") as f:
            json.dump({"args": vars(args), "test_loss": tl, "test_acc": ta, "history": hist}, f)