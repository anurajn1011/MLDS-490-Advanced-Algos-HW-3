"""
val_differential_privacy.py - edge cases and validation experiments for Part 2 (Laplace input perturbation).
Every check is tagged [EDGE] (edge case or limit case) or [VALID] (validation of correct behavior).
Run from src/validation:  python validation_dp.py (QUICK=1 for a short run)

Blocks: the noise really is Laplace(0, b) and independent; the design rules (fixed noise, clean test set, b=0 reproduces Part 1) hold in the real code path; the noise power matches theory; an attacker recovering pixels from the perturbed data does exactly as well as Laplace theory predicts (so b means what we claim); and accuracy falls with b once the learning rate confound is removed.
"""
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from val_common import add_paths, load_data, check, skip, section, finish, QUICK, FIG
add_paths("q2", "q1/fed_avg")
from model import pre_process
from clients import Client
from server import Server
from noisy_client import NoisyClient, perturb
from run_dp import run as dp_run

torch.set_num_threads(1)
train, test = load_data()
N = len(train)
XT, YT = test[0]["images"], test[0]["labels"]
LR, E, BS = 0.05, 3, 32
RES = {}

def pair(i, b, seed=0, clip=False):
    c = Client(i, train[i]["images"], train[i]["labels"], seed=seed)
    n = NoisyClient(i, train[i]["images"], train[i]["labels"], noise_b=b, seed=seed, clip=clip)
    return c, n

def eps_of(i, b, seed=0):
    c, n = pair(i, b, seed)
    return (n.X_train - c.X_train).double(), (n.X_val - c.X_val).double(), c, n

def ks(x, cdf):
    x = np.sort(x); n = len(x); F = cdf(x)
    return max((np.arange(1, n + 1) / n - F).max(), (F - np.arange(0, n) / n).max())

def corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])

### D0
def distribution_tests():
    section("D0 THE INJECTED NOISE IS LAPLACE(0, b)")
    b = 0.5
    parts = [eps_of(i, b) for i in range(10)]
    eps = torch.cat([torch.cat([p[0].flatten(), p[1].flatten()]) for p in parts]).numpy()
    n = len(eps); mu = eps.mean()
    check("[VALID] D0 sample mean is 0 (within 5 standard errors)", abs(mu) < 5 * math.sqrt(2) * b / math.sqrt(n), f"mean = {mu:.2e}")
    check("[VALID] D0 E|eps| = b (within 1%)", abs(np.abs(eps).mean() / b - 1) < 0.01, f"{np.abs(eps).mean() / b:.4f}")
    check("[VALID] D0 Var(eps) = 2 b^2 (within 3%)", abs(eps.var() / (2 * b * b) - 1) < 0.03, f"{eps.var() / (2 * b * b):.4f}")
    kurt = ((eps - mu) ** 4).mean() / eps.var() ** 2 - 3
    check("[VALID] D0 excess kurtosis is 3 (Laplace; a Gaussian would give 0)", abs(kurt - 3) < 0.3, f"{kurt:.3f}")

### D1
def independence_tests():
    section("D1 NOISE IS INDEPENDENT ACROSS PIXELS, IMAGES, CLIENTS AND THE SIGNAL")
    e0, _, c0, _ = eps_of(0, 1.0)
    e1, _, _, _ = eps_of(1, 1.0)
    m = min(len(e0), len(e1))
    img = e0.reshape(-1, 28, 28).numpy()
    check("[VALID] D1 noise is uncorrelated with the clean pixel value", abs(corr(e0, c0.X_train)) < 0.01, f"corr = {corr(e0, c0.X_train):.4f}")
    check("[VALID] D1 horizontally neighbouring pixels get independent noise", abs(corr(img[:, :, :-1], img[:, :, 1:])) < 0.01)
    check("[VALID] D1 two clients get independent noise (separate RNG streams)", abs(corr(e0[:m], e1[:m])) < 0.01)
    check("[VALID] D1 two images of the same client get independent noise", abs(corr(e0[0], e0[1])) < 0.15)
    check("[VALID] D1 no image reuses another image's noise pattern", len({tuple(r[:6].tolist()) for r in e0}) == len(e0))

### D2
def design_tests():
    section("D2 DESIGN RULES: NOISE IS FIXED, SCALES LINEARLY IN b, SPLIT AND LABELS UNTOUCHED")
    c, n = pair(2, 0.7)
    X0 = n.X_train.clone()
    g = {k: v.clone() for k, v in Server([n], XT, YT, C=1.0, seed=0).model.state_dict().items()}
    n.local_update(g, 2, LR, BS, seed=1); n.local_update(g, 2, LR, BS, seed=2)
    check("[VALID] D2 the perturbed dataset is fixed: local training never resamples or alters it", torch.equal(n.X_train, X0))
    check("[VALID] D2 rebuilding with the same seed gives the same noise, another seed a different one",
          torch.equal(pair(2, 0.7)[1].X_train, X0) and not torch.equal(pair(2, 0.7, seed=1)[1].X_train, X0))
    ea, eb = eps_of(2, 0.5)[0], eps_of(2, 2.0)[0]
    check("[VALID] D2 noise at b=2 is exactly 4x the noise at b=0.5 (common random numbers across b)", torch.allclose(eb / 4, ea, atol=2e-5))
    check("[VALID] D2 the train/val split is identical to the clean client's (noise changes values, not membership)",
          torch.equal(c.y_train, n.y_train) and torch.equal(c.y_val, n.y_val))
    ev = (n.X_val - c.X_val).abs().mean().item()
    check("[VALID] D2 the validation images are perturbed too (mean |change| = b)", abs(ev / 0.7 - 1) < 0.1, f"{ev:.3f} vs 0.7")

### D3
def isolation_tests():
    section("D3 TEST SET IS CLEAN; b = 0 REPRODUCES PART 1 EXACTLY")
    cl = [NoisyClient(i, train[i]["images"], train[i]["labels"], noise_b=2.0, seed=0) for i in range(20)]
    srv = Server(cl, XT, YT, C=0.1, seed=0)
    srv.train(2, 1, 0.005, BS)
    check("[VALID] D3 after training on noisy clients the server's test tensor equals the clean preprocessed test set",
          torch.equal(srv.X_test, torch.from_numpy(pre_process(XT))))
    c, n0 = pair(5, 0.0)
    check("[EDGE] D3 b = 0 leaves every pixel exactly unchanged", torch.equal(c.X_train, n0.X_train) and torch.equal(c.X_val, n0.X_val))
    ref = Server([Client(i, train[i]["images"], train[i]["labels"], seed=0) for i in range(N)], XT, YT, C=0.1, seed=0).train(3, 1, LR, BS)
    h, *_ = dp_run(train, test, 0.0, C=0.1, epochs=1, lr=LR, batch_size=BS, rounds=3, seed=0)
    check("[EDGE] D3 a full FedAvg run with b = 0 has exactly the Part 1 history", all(a["train_loss"] == r["train_loss"] and a["val_acc"] == r["val_acc"] for a, r in zip(h, ref)))

### D4
def power_and_edge_tests():
    section("D4 NOISE POWER, CLIPPING AND EXTREME b")
    for b in (0.1, 0.5, 1.0):
        ps = [pair(i, b)[1] for i in range(30)]
        cs = [pair(i, 0.0)[1] for i in range(30)]
        meas = np.mean([p.noisy_sq for p in ps]); clean = np.mean([c.clean_sq for c in cs])
        pred = clean + 2 * 784 * b * b
        check(f"[VALID] D4 b={b}: mean ||x'||^2 = ||x||^2 + 2 * 784 * b^2 (within 2%)", abs(meas / pred - 1) < 0.02, f"{meas:.1f} vs {pred:.1f}")
    x = pair(0, 0.0)[0].X_train[:100]
    a = perturb(x, 1.0, np.random.default_rng(0)); c = perturb(x, 1.0, np.random.default_rng(0), clip=True)
    check("[EDGE] D4 clipping keeps pixels in [0, 1] and moves the image less than raw noise",
          c.min() >= 0 and c.max() <= 1 and (c - x).pow(2).mean() < (a - x).pow(2).mean())
    try:
        perturb(x, -1.0, np.random.default_rng(0)); raised = False
    except ValueError:
        raised = True
    check("[EDGE] D4 a negative scale is rejected", raised)
    check("[EDGE] D4 b = 1e-12 is indistinguishable from no noise", (perturb(x, 1e-12, np.random.default_rng(0)) - x).abs().max() < 1e-6)
    big = perturb(x, 1e6, np.random.default_rng(0))
    check("[EDGE] D4 b = 1e6 stays finite float32 with the right shape", torch.isfinite(big).all() and big.dtype == torch.float32 and big.shape == x.shape)

### D5
def centroid_probe():
    section("D5 DATA-LEVEL INFORMATION LOSS (no learning rate, no optimizer involved)")
    clean = [Client(i, train[i]["images"], train[i]["labels"], seed=0) for i in range(N)]
    X = torch.cat([c.X_train for c in clean]); y = torch.cat([c.y_train for c in clean])
    cent = torch.stack([X[y == k].mean(0) if (y == k).any() else torch.full((784,), 1e3) for k in range(62)])
    accs = {}
    for b in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
        v = [NoisyClient(i, train[i]["images"], train[i]["labels"], noise_b=b, seed=0) for i in range(N)]
        Xv = torch.cat([c.X_val for c in v]); yv = torch.cat([c.y_val for c in v])
        accs[b] = (torch.cdist(Xv, cent).argmin(1) == yv).float().mean().item()
    print("\tnearest-centroid accuracy on perturbed validation images:", {k: round(v, 3) for k, v in accs.items()})
    vals = list(accs.values())
    check("[VALID] D5 nearest-centroid accuracy never rises as b grows (tolerance 0.01)", all(b2 <= b1 + 0.01 for b1, b2 in zip(vals, vals[1:])))
    check("[VALID] D5 at b = 5 the data has lost at least 0.1 of the b = 0 accuracy", accs[5.0] < accs[0.0] - 0.1)

### D6
def attack_tests():
    section("D6 PIXEL-RECOVERY ATTACK MATCHES LAPLACE THEORY (does b mean what we claim?)")
    bs = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
    emp, th, bound = [], [], []
    for b in bs:
        bg, ink = [], []
        for i in range(20):
            c, n = pair(i, b)
            cv, nv = c.X_train.flatten(), n.X_train.flatten()
            bg.append(nv[cv == 0]); ink.append(nv[cv >= 0.999])
        bg, ink = torch.cat(bg), torch.cat(ink)
        k = min(len(bg), len(ink))
        if k < 5000:
            skip("D6 attack accuracy", f"only {k} exact-0 / exact-1 pixels"); return
        acc = 0.5 * ((bg[:k] <= 0.5).float().mean() + (ink[:k] > 0.5).float().mean()).item()
        t = 1 - 0.5 * math.exp(-1 / (2 * b)); eps = 1 / b
        emp.append(acc); th.append(t); bound.append(math.exp(eps) / (1 + math.exp(eps)))
    print("\tattack accuracy, empirical vs theory:", [(b, round(e, 4), round(t, 4)) for b, e, t in zip(bs, emp, th)])
    check("[VALID] D6 empirical pixel-recovery accuracy matches 1 - 0.5 exp(-1/(2b)) within 0.005", max(abs(e - t) for e, t in zip(emp, th)) < 0.005)
    check("[VALID] D6 no attacker beats the epsilon = 1/b differential-privacy bound e^eps / (1 + e^eps)", all(e <= bd + 0.002 for e, bd in zip(emp, bound)))
    check("[VALID] D6 attack accuracy falls toward 0.5 (guessing) as b grows", all(a >= c for a, c in zip(emp, emp[1:])) and emp[-1] < 0.6)
    RES["attack"] = (bs, emp, th, bound)

### D7 / D8
def utility_tests():
    section("D7 LEARNING-RATE CONFOUND AND D8 ACCURACY VS b")
    R = 20 if QUICK else 60
    seeds = (0,) if QUICK else (0, 1)
    bs = (0.0, 0.25, 0.5, 1.0, 2.0)
    scaled = {b: [dp_run(train, test, b, C=0.1, epochs=E, lr=LR, batch_size=BS, rounds=R, seed=s, scale_lr=True) for s in seeds] for b in bs}
    fixed = {b: dp_run(train, test, b, C=0.1, epochs=E, lr=LR, batch_size=BS, rounds=R, seed=0, scale_lr=False) for b in (0.5, 2.0)}
    print(f"\n\t{'b':>5}{'lr used':>9}{'test acc (scaled lr)':>22}{'test acc (fixed lr)':>21}")
    for b in bs:
        ta = [r[2] for r in scaled[b]]
        fx = f"{fixed[b][2]:.3f}{' (diverged)' if not np.isfinite(fixed[b][1]) else ''}" if b in fixed else "-"
        print(f"\t{b:>5}{scaled[b][0][3]:>9.4f}{np.mean(ta):>14.3f} +- {np.std(ta):.3f}{fx:>21}")
    for b in (0.5, 2.0):
        check(f"[VALID] D7 b={b}: the scaled learning rate is smaller than the fixed one", scaled[b][0][3] < LR)
        check(f"[VALID] D7 b={b}: scaling the learning rate never hurts test accuracy (tolerance 0.01)", np.mean([r[2] for r in scaled[b]]) >= fixed[b][2] - 0.01, f"scaled {np.mean([r[2] for r in scaled[b]]):.3f}, fixed {fixed[b][2]:.3f}")
    check("[VALID] D7 with the scaled learning rate no run diverges", all(np.isfinite(r[1]) for b in bs for r in scaled[b]))
    acc = [np.mean([r[2] for r in scaled[b]]) for b in bs]
    check("[VALID] D8 mean test accuracy never rises as b grows (tolerance 0.02)", all(b2 <= b1 + 0.02 for b1, b2 in zip(acc, acc[1:])))
    check("[VALID] D8 b = 2 loses at least 0.1 of the noiseless test accuracy", acc[-1] < acc[0] - 0.1, f"{acc[0]:.3f} -> {acc[-1]:.3f}")
    check("[VALID] D8 noise never helps: every noisy test accuracy is at most the noiseless one + 0.02", all(a <= acc[0] + 0.02 for a in acc))
    train_acc = [np.mean([np.mean([h["train_acc"] for h in r[0][-10:]]) for r in scaled[b]]) for b in bs]
    RES["utility"] = (bs, acc, train_acc)

def privacy_utility_plot():
    if "attack" not in RES or "utility" not in RES:
        return
    ab, emp, th, bound = RES["attack"]
    ub, acc, tr = RES["utility"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ab, emp, "o", label="empirical attacker")
    ax[0].plot(ab, th, "-", label="theory 1 - 0.5 exp(-1/2b)")
    ax[0].plot(ab, bound, "--", label="DP bound e^eps/(1+e^eps), eps=1/b")
    ax[0].set(xscale="log", xlabel="noise scale b", ylabel="attacker accuracy (per pixel)", title="privacy side")
    ax[0].legend(fontsize=8)
    att = [1.0 if b == 0 else 1 - 0.5 * math.exp(-1 / (2 * b)) for b in ub]
    ax[1].plot(att, acc, "o-", label="test accuracy")
    ax[1].plot(att, tr, "s--", label="training accuracy (noisy data)")
    for b, x, y in zip(ub, att, acc):
        ax[1].annotate(f"b={b:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax[1].invert_xaxis()
    ax[1].set(xlabel="attacker accuracy (left = more private)", ylabel="accuracy", title="privacy-utility frontier")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "val_dp_privacy_utility.png", dpi=300)
    plt.close()
    print("\tsaved", FIG / "val_dp_privacy_utility.png")

def main():
    distribution_tests()
    independence_tests()
    design_tests()
    isolation_tests()
    power_and_edge_tests()
    centroid_probe()
    attack_tests()
    utility_tests()
    privacy_utility_plot()
    finish()


if __name__ == "__main__":
    main()