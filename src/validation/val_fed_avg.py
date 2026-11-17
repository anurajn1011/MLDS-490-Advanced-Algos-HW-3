"""
val_fed_avg.py - edge cases and validation experiments for Part 1.1 (sequential FedAvg).
Every check is tagged [EDGE] (edge case or limit case) or [VALID] (validation of correct behavior).
Run from src/validation:  python validation_fedavg.py, QUICK=1 for a short run

Tests an exact identity, a limit case, a control experiment with a known answer, a leakage probe, etc.
"""
import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from val_common import (add_paths, load_data, check, section, finish, flat, max_diff, QUICK, FIG)
add_paths("q1/fed_avg")
from model import MLP, pre_process
from clients import Client
from server import Server
from non_standard_plot import run_recording, mean_cosine

torch.set_num_threads(1)
train_data, test_data = load_data()
N, SEED = len(train_data), 0
XT, YT = test_data[0]["images"], test_data[0]["labels"]

### HELPERS ###
class Stub:
    def __init__(self, i, n):
        self.id, self.n_train = i, n

def make_clients(seed=SEED):
    return [Client(i, train_data[i]["images"], train_data[i]["labels"], seed=seed) for i in range(N)]

def make_server(clients, C=0.1, seed=SEED):
    return Server(clients, XT, YT, C=C, seed=seed)

def fit(clients, rounds, E, lr=0.05, C=0.1, m=None, seed=SEED):
    srv = make_server(clients, C, seed)
    return srv, srv.train(rounds, E, lr, 32, m=m)

def tail(h, key, k=10):
    return float(np.mean([x[key] for x in h[-k:]]))

def pooled_gd(init, clients, steps, lr):
    """Centralized full-batch gradient descent on the union of all clients training data."""
    m = MLP()
    m.load_state_dict(init)
    X = torch.cat([c.X_train for c in clients]); y = torch.cat([c.y_train for c in clients])
    for _ in range(steps):
        m.zero_grad()
        nn.functional.cross_entropy(m(X), y).backward()
        with torch.no_grad():
            for p in m.parameters():
                p -= lr * p.grad # Manual SGD
    return m.state_dict()
######

### F0
def data_tests():
    section("F0 DATA, LEAKAGE AND PREPROCESSING")
    imgs = [np.asarray(train_data[i]["images"]) for i in range(N)]
    labs = np.concatenate([np.asarray(train_data[i]["labels"]).astype(int) for i in range(N)])
    check("[VALID] F0 labels lie in [0, 61] for train and test",
          labs.min() >= 0 and labs.max() <= 61 and np.asarray(YT).min() >= 0 and np.asarray(YT).max() <= 61)
    check("[VALID] F0 every client has as many labels as images", all(len(imgs[i]) == len(train_data[i]["labels"]) for i in range(N)))
    h = lambda a: {hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest() for x in a}
    tr = set().union(*[h(x) for x in imgs])
    overlap = len(tr & h(XT))
    check("[VALID] F0 no test image appears in any client's training data (leakage probe)", overlap == 0, f"{overlap} shared images")
    dup = sum(len(h(x)) for x in imgs) - len(tr)
    print(f"\tnote: {dup} images are duplicated across clients (clients are supposed to hold disjoint data)")
    white = np.ones((3, 28, 28), np.float32)
    p = pre_process(white)
    check("[VALID] F0 pre_process maps the white background to 0 and flattens to 784", p.shape == (3, 784) and p.max() == 0.0)
    pm = pre_process(imgs[0])
    check("[EDGE] F0 after pre_process most pixels are background (mean pixel < 0.3), the cause of the lr=0.1 blow-up we saw earlier",
          pm.mean() < 0.3, f"mean pixel = {pm.mean():.3f}")

### F1
def split_tests():
    section("F1 LOCAL 80/20 SPLIT")
    for n in (118, 331, 393):
        X = np.zeros((n, 28, 28), np.float32); X[:, 0, 0] = (np.arange(n) + 1) / (n + 1)
        y = np.arange(n) % 62
        key = {round(float(v), 6): i for i, v in enumerate(pre_process(X)[:, 0])}
        c = Client(3, X, y, seed=0)
        tr = [key[round(float(v), 6)] for v in c.X_train[:, 0]]
        va = [key[round(float(v), 6)] for v in c.X_val[:, 0]]
        check(f"[VALID] F1 n={n}: train and validation are disjoint and together cover all samples",
              set(tr).isdisjoint(va) and set(tr) | set(va) == set(range(n)))
        check(f"[VALID] F1 n={n}: validation has int(0.2 n) = {int(0.2 * n)} samples", len(va) == int(0.2 * n))
        check(f"[VALID] F1 n={n}: labels stay attached to their images after the shuffle",
              np.array_equal(c.y_train.numpy(), np.array(tr) % 62) and np.array_equal(c.y_val.numpy(), np.array(va) % 62))
    X = np.zeros((200, 28, 28), np.float32); X[:, 0, 0] = np.linspace(0, 1, 200); y = np.arange(200) % 62
    a, b = Client(0, X, y, seed=0), Client(0, X, y, seed=0)
    c, d = Client(0, X, y, seed=1), Client(1, X, y, seed=0)
    check("[VALID] F1 the same seed gives the same split", torch.equal(a.X_val, b.X_val))
    check("[VALID] F1 a different seed or a different client id gives a different split",
          not torch.equal(a.X_val, c.X_val) and not torch.equal(a.X_val, d.X_val))
    tiny = Client(0, X[:3], y[:3], seed=0)
    sd, n, met = tiny.local_update(MLP().state_dict(), 2, 0.05, 32)
    check("[EDGE] F1 edge case n=3: empty validation set and batch_size > n do not crash",
          tiny.n_val == 0 and met["val"][2] == 0 and n == 3 and all(torch.isfinite(v).all() for v in sd.values()))

### F2
def aggregation_tests():
    section("F2 WEIGHTED AGGREGATION")
    keys = list(MLP().state_dict())
    const = lambda c: {k: torch.full_like(v, c) for k, v in MLP().state_dict().items()}
    srv = Server([Stub(0, 100), Stub(1, 300)], XT, YT, C=1.0, seed=0)
    srv.aggregate([0, 1], [const(1.0), const(5.0)])
    check("[VALID] F2 100 vs 300 samples: result is 0.25 * 1 + 0.75 * 5 = 4", all(torch.allclose(v, torch.full_like(v, 4.0)) for v in srv.model.state_dict().values()))
    srv.aggregate([0, 1], [const(2.0), const(2.0)])
    check("[VALID] F2 weights sum to one: averaging identical models returns that model", all(torch.allclose(v, torch.full_like(v, 2.0)) for v in srv.model.state_dict().values()))

    ns = [50, 120, 330]
    torch.manual_seed(0)
    states = [{k: torch.randn_like(v) for k, v in MLP().state_dict().items()} for _ in ns]
    srv = Server([Stub(i, n) for i, n in enumerate(ns)], XT, YT, C=1.0, seed=0)
    snapshot = [{k: v.clone() for k, v in s.items()} for s in states]
    srv.aggregate([0, 1, 2], states)
    ref = {k: sum(states[i][k].double() * ns[i] / sum(ns) for i in range(3)) for k in keys}
    got = srv.model.state_dict()
    check("[VALID] F2 random states match a float64 reference average", all(torch.allclose(got[k].double(), ref[k], atol=1e-6) for k in keys))
    srv2 = Server([Stub(i, n) for i, n in enumerate(ns)], XT, YT, C=1.0, seed=0)
    srv2.aggregate([2, 0, 1], [states[2], states[0], states[1]])
    check("[VALID] F2 the order in which clients are listed does not change the result", max_diff(srv2.model.state_dict(), got) < 1e-6)
    check("[VALID] F2 aggregation does not modify the clients' state dicts", all(torch.equal(states[i][k], snapshot[i][k]) for i in range(3) for k in keys))
    srv3 = Server([Stub(i, 100) for i in range(3)], XT, YT, C=1.0, seed=0)
    srv3.aggregate([0, 1, 2], states)
    ref_eq = {k: sum(s[k] for s in states) / 3 for k in keys}
    check("[EDGE] F2 equal sizes reduce to the plain mean", all(torch.allclose(srv3.model.state_dict()[k], ref_eq[k], atol=1e-6) for k in keys))

### F3
def selection_tests():
    section("F3 CLIENT SELECTION")
    stubs = [Stub(i, 100 + 3 * i) for i in range(100)]
    for C in (0.02, 0.05, 0.10):
        s = Server(stubs, XT, YT, C=C, seed=0)
        ids = s.select_clients()
        check(f"[EDGE] F3 C={C} selects {round(C * 100)} distinct clients", len(ids) == round(C * 100) and len(set(ids)) == len(ids), f"got {len(ids)}")
    s = Server(stubs, XT, YT, C=0.1, seed=1)
    R, counts = 3000, np.zeros(100)
    for _ in range(R):
        counts[s.select_clients()] += 1
    z = np.abs(counts - R * 0.1) / math.sqrt(R * 0.1 * 0.9)
    check("[VALID] F3 selection is uniform: no client is over- or under-selected by more than 4.5 sigma", z.max() < 4.5, f"max z = {z.max():.2f}")
    corr = np.corrcoef(counts, [c.n_train for c in stubs])[0, 1]
    check("[VALID] F3 selection does not depend on a client's data size", abs(corr) < 0.35, f"corr = {corr:.2f}")
    a, b, c = (Server(stubs, XT, YT, C=0.1, seed=sd).select_clients() for sd in (4, 4, 5))
    check("[VALID] F3 same seed repeats the selection, different seed changes it", a == b and a != c)
    for bad in (0, 101):
        try:
            Server(stubs, XT, YT, C=0.1, seed=0).select_clients(bad); ok = False
        except Exception:
            ok = True
        check(f"[EDGE] F3 m={bad} is rejected", ok)

### F4
def client_tests(clients):
    section("F4 LOCAL UPDATE")
    c = clients[0]
    torch.manual_seed(0)
    g = MLP().state_dict()
    snap = {k: v.clone() for k, v in g.items()}
    new, n, met = c.local_update(g, 3, 0.05, 32, seed=1)
    check("[VALID] F4 local_update does not modify the global weights it received", all(torch.equal(g[k], snap[k]) for k in g))
    check("[VALID] F4 it returns n_train and metric counts that match the data", n == c.n_train and met["train"][2] == c.n_train and met["val"][2] == c.n_val and met["train"][1] <= c.n_train)
    check("[EDGE] F4 epochs = 0 returns the global weights unchanged", max_diff(c.local_update(g, 0, 0.05, 32)[0], g) == 0.0)
    check("[EDGE] F4 lr = 0 returns the global weights unchanged for any E", max_diff(c.local_update(g, 4, 0.0, 32)[0], g) == 0.0)
    m0 = MLP(); m0.load_state_dict(g)
    l0 = c._evaluate(m0, c.X_train, c.y_train)[0] / c.n_train
    l1 = c.local_update(g, 5, 0.1, 32)[2]["train"][0] / c.n_train
    check("[VALID] F4 five local epochs lower the client's training loss", l1 < l0, f"{l0:.3f} -> {l1:.3f}")
    s = {k: v.clone() for k, v in g.items()}
    for _ in range(40):
        s, _, met2 = c.local_update(s, 5, 0.1, 32)
    acc = met2["train"][1] / met2["train"][2]
    check("[VALID] F4 a single client can overfit its own data (train accuracy > 0.9), so model, loss and optimizer work", acc > 0.9, f"acc = {acc:.3f}")
    import copy
    c2 = copy.copy(c); c2.y_val = (c.y_val + 1) % 62
    check("[VALID] F4 validation labels never influence the local weights (scrambling them changes nothing)",
          max_diff(c.local_update(g, 2, 0.05, 32, seed=7)[0], c2.local_update(g, 2, 0.05, 32, seed=7)[0]) == 0.0)

### F5
def equivalence_tests(clients):
    section("F5 EXACT IDENTITY: ONE FULL-BATCH ROUND OF FEDAVG = ONE STEP OF CENTRALIZED GD")
    lr = 0.1
    gaps = {}
    for E in (1, 2, 5, 10):
        srv = make_server(clients, C=1.0)
        init = {k: v.clone() for k, v in srv.model.state_dict().items()}
        srv.train(1, E, lr, 10 ** 6, m=N)
        gaps[E] = max_diff(srv.model.state_dict(), pooled_gd(init, clients, E, lr))
    print("\tmax |FedAvg - centralized GD| by E:", {k: f"{v:.2e}" for k, v in gaps.items()})
    check("[VALID] F5 E=1, full batch, all clients: FedAvg equals centralized gradient descent (weights agree to 1e-5)", gaps[1] < 1e-5, f"gap = {gaps[1]:.2e}")
    check("[VALID] F5 for E > 1 the two differ and the gap grows with E (client drift on non-IID data)",
          gaps[1] < gaps[2] < gaps[5] < gaps[10] and gaps[10] > 100 * max(gaps[1], 1e-9), f"gap(E=10) = {gaps[10]:.2e}")
    plt.figure(figsize=(5, 3.5))
    plt.semilogy(list(gaps), [max(v, 1e-9) for v in gaps.values()], "o-")
    plt.xlabel("local epochs E"); plt.ylabel("max |FedAvg - centralized GD|"); plt.title("client drift after one round")
    plt.tight_layout()
    plt.savefig(FIG / "val_fed_avg_drift_vs_E.png", dpi=300)
    plt.close()

### F6
def variance_tests(clients):
    section("F6 SAMPLING VARIANCE OF THE AGGREGATED UPDATE SHRINKS LIKE 1/m")
    torch.manual_seed(0)
    g = MLP().state_dict()
    gv = flat(g)
    D = torch.stack([flat(c.local_update(g, 3, 0.05, 32, seed=1)[0]) - gv for c in clients])
    w = torch.tensor([c.n_train for c in clients], dtype=torch.float32)
    rng = np.random.default_rng(0)
    full = (w[:, None] * D).sum(0) / w.sum()
    out = {}
    for m in (2, 5, 10, 20, 50):
        err = []
        for _ in range(400):
            ids = rng.choice(N, m, replace=False)
            ww = w[ids] / w[ids].sum()
            err.append(((ww[:, None] * D[ids]).sum(0) - full).pow(2).sum().item())
        out[m] = float(np.mean(err))
    unit = {m: out[m] * m / (1 - (m - 1) / (N - 1)) for m in out}
    ratio = max(unit.values()) / min(unit.values())
    print("\tE|aggregate - full-participation update|^2 :", {m: f"{v:.3e}" for m, v in out.items()})
    check("[VALID] F6 variance x m / finite-population factor is roughly constant across m (within 2.5x)", ratio < 2.5, f"max/min = {ratio:.2f}")
    check("[VALID] F6 variance falls monotonically as m grows (why m=4 curves are noisier)", all(out[a] > out[b] for a, b in ((2, 5), (5, 10), (10, 20), (20, 50))))

### F7
def noniid_tests(clients):
    section("F7 NON-IID CONTROL: SAME DATA, SAME SIZES, LABELS REDISTRIBUTED IID")
    X = np.concatenate([np.asarray(train_data[i]["images"]) for i in range(N)])
    y = np.concatenate([np.asarray(train_data[i]["labels"]) for i in range(N)])
    sizes = [len(train_data[i]["labels"]) for i in range(N)]
    perm = np.random.default_rng(0).permutation(len(X))
    iid = [Client(i, X[s], y[s], seed=SEED) for i, s in enumerate(np.split(perm, np.cumsum(sizes)[:-1]))]
    R, E = (30 if QUICK else 300), 5
    rec_n = run_recording(make_server(clients), {c.id: c for c in clients}, R, E, 0.05, 32)
    rec_i = run_recording(make_server(iid), {c.id: c for c in iid}, R, E, 0.05, 32)
    an, ai = np.mean(rec_n["val_acc"][-10:]), np.mean(rec_i["val_acc"][-10:])
    cn, ci = mean_cosine(rec_n), mean_cosine(rec_i)
    print(f"\tval acc: non-IID {an:.3f}, IID {ai:.3f} | mean update cosine: non-IID {cn:.3f}, IID {ci:.3f}")
    check("[VALID] F7 clients' updates agree more when the data is IID (higher mean cosine)", ci > cn)
    check("[VALID] F7 FedAvg is no better on the real non-IID split than on the IID split (within 0.02)", an <= ai + 0.02)

### F8
def learning_tests(clients):
    section("F8 LEARNING AGAINST KNOWN FLOORS")
    labs = np.concatenate([np.asarray(train_data[i]["labels"]).astype(int) for i in range(N)])
    maj = np.bincount(labs, minlength=62).max() / len(labs)
    fresh = make_server(clients)
    l0, a0 = fresh.evaluate_test()
    check("[VALID] F8 untrained model has test loss near ln 62 = 4.13", 3.9 < l0 < 4.6, f"loss = {l0:.3f}")
    R = 40 if QUICK else 120
    srv, h = fit(clients, R, 5)
    va = tail(h, "val_acc")
    check(f"[VALID] F8 validation accuracy after {R} rounds is far above the majority-class rate ({maj:.3f})", va > 3 * maj, f"val acc = {va:.3f}")
    check("[VALID] F8 training loss decreases (mean of the last 10 rounds < first 10)", tail(h, "train_loss") < np.mean([x["train_loss"] for x in h[:10]]))
    check("[VALID] F8 no round produced a non-finite loss", all(np.isfinite(x["train_loss"]) and np.isfinite(x["val_loss"]) for x in h))
    rng = np.random.default_rng(3)
    shuf = [Client(i, train_data[i]["images"], rng.permutation(np.asarray(train_data[i]["labels"])), seed=SEED) for i in range(N)]
    _, hs = fit(shuf, R, 5)
    vs = tail(hs, "val_acc")
    check("[VALID] F8 control: with shuffled labels validation accuracy stays near the label-marginal floor", vs <= maj + 0.05, f"{vs:.3f} vs floor {maj:.3f}")
    return h

### F9
def control_tests(clients):
    section("F9 TEST DATA ISOLATION AND DETERMINISM")
    srv = make_server(clients)
    srv.X_test = torch.full_like(srv.X_test, float("nan"))
    h = srv.train(3, 1, 0.05, 32)
    check("[VALID] F9 training never touches the test set (a NaN-poisoned test set leaves training finite)",
          all(np.isfinite(x["train_loss"]) for x in h) and all(torch.isfinite(p).all() for p in srv.model.parameters()))
    check("[VALID] F9 ... and evaluate_test does read it (the poison is real)", math.isnan(srv.evaluate_test()[0]))
    a, b, c = (fit(clients, 3, 2, seed=s)[0].model.state_dict() for s in (0, 0, 1))
    check("[VALID] F9 the same seed reproduces the weights exactly", max_diff(a, b) == 0.0)
    check("[VALID] F9 a different seed changes them", max_diff(a, c) > 1e-4)

def main():
    clients = make_clients()
    data_tests()
    split_tests()
    aggregation_tests()
    selection_tests()
    client_tests(clients)
    equivalence_tests(clients)
    variance_tests(clients)
    noniid_tests(clients)
    learning_tests(clients)
    control_tests(clients)
    finish()

if __name__ == "__main__":
    main()