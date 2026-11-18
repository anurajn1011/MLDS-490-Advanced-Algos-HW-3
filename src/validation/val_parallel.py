"""
val_parallel.py - edge cases and validation experiments for Part 1.2 (Ray actors).
Every check is tagged [EDGE] (edge case or limit case) or [VALID] (validation of correct behavior).
Run from src/validation:  python validation_parallel.py (QUICK=1 for a short run)
Recommended to run using script: scripts/run_validation.sh parallel

Uses the first 20 clients so that every test is fast. Needs at least 4 CPUs for the speedup checks otherwise these checks are skipped. 
"""
import os
import threading
import time

import numpy as np
import torch

from val_common import (add_paths, load_data, check, skip, section, finish, max_diff, flat, QUICK)
add_paths("q1/parallel_clients", "q1/fed_avg")
import ray
import paths
from model import MLP
from clients import Client
from server import Server
from parallel_server import ParallelServer
from client_actor import ClientActor

torch.set_num_threads(1)
train_data, test_data = load_data()
K = 20
SUB = train_data[:K]
XT, YT = test_data[0]["images"], test_data[0]["labels"]
E, LR, BS = 2, 0.05, 32

def seq_server():
    cl = [Client(i, SUB[i]["images"], SUB[i]["labels"], seed=0) for i in range(K)]
    return Server(cl, XT, YT, C=0.1, seed=0), cl

def par_server():
    return ParallelServer(SUB, XT, YT, seed=0)

def actor_procs():
    import psutil
    n = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if "ClientActor" in " ".join(p.info.get("cmdline") or []) + (p.info.get("name") or ""):
                n += 1
        except Exception:
            pass
    return n

def wait_no_actors(timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        if actor_procs() == 0:
            return True
        time.sleep(0.2)
    return False

### P1
def equivalence_tests():
    section("P1 PARALLEL = SEQUENTIAL")
    seq, cl = seq_server()
    par = par_server()
    check("[VALID] P1 the server's client table (n_train) matches the real 80/20 split of every client",
          all(par.n_train[i] == cl[i].n_train for i in range(K)))
    check("[VALID] P1 the parallel server holds only (id, n_train) stubs, never client data", set(par.clients[0]._fields) == {"id", "n_train"})
    R = 3 if QUICK else 5
    hs, hp = seq.train(R, E, LR, BS, m=4), par.train(R, E, LR, BS, m=4)
    check("[VALID] P1 both runs sample exactly the same clients in every round", [h["clients"] for h in hs] == [h["clients"] for h in hp])
    d = max_diff(seq.model.state_dict(), par.model.state_dict())
    check(f"[VALID] P1 weights after {R} rounds agree (max |diff| < 1e-4; actors use 1 thread, so not bit-identical)", d < 1e-4, f"max diff = {d:.2e}")
    check("[VALID] P1 the logged train/val loss and accuracy agree",
          all(abs(a[k] - b[k]) < 1e-3 for a, b in zip(hs, hp) for k in ("train_loss", "val_loss", "train_acc", "val_acc")))
    cpus = int(ray.cluster_resources().get("CPU", 0))
    check("[EDGE] P1 the Ray cluster has at least m = 4 CPUs (an actor keeps its CPU while alive, so fewer CPUs would hang the round, not queue it)", cpus >= 4, f"{cpus} CPUs")
    for m in sorted({1, min(8, cpus)}):  # one actor; as many actors as CPUs
        s, _ = seq_server(); p = par_server()
        s.train(1, E, LR, BS, m=m); p.train(1, E, LR, BS, m=m)
        d = max_diff(s.model.state_dict(), p.model.state_dict())
        check(f"[EDGE] P1 edge case m={m}: one round still matches the sequential result", d < 1e-4, f"max diff = {d:.2e}")

### P2
def lifecycle_tests():
    section("P2 ACTORS EXIST ONLY FOR THE SELECTED CLIENTS, AND ARE FREED")
    try:
        import psutil  # noqa: F401
    except ImportError:
        skip("P2 actor process counts", "psutil not installed"); return
    peak, stop = [0], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], actor_procs()); time.sleep(0.05)

    th = threading.Thread(target=sample, daemon=True); th.start()
    par_server().train(3, 5, LR, BS, m=4)
    stop.set(); th.join()
    check("[VALID] P2 never more than m = 4 actor processes alive at once (no 100-actor pool)", peak[0] <= 4, f"peak = {peak[0]}")
    check("[VALID] P2 at least 2 actors were alive at the same time (the clients really overlap)", peak[0] >= 2, f"peak = {peak[0]}")
    check("[VALID] P2 all actors are gone after training (ray.kill works)", wait_no_actors())

### P3
def resource_tests():
    section("P3 RESOURCE CAP AND DATA ISOLATION INSIDE AN ACTOR")
    a = ClientActor.options(num_cpus=1, num_gpus=0).remote(7, SUB[7]["images"], SUB[7]["labels"], 0.2, 0)
    try:
        info = ray.get(a.__ray_call__.remote(lambda self: (ray.get_runtime_context().get_assigned_resources(),
                                                           torch.get_num_threads(), self.client.id, self.client.n_train)))
    except Exception as e:  # older Ray without __ray_call__
        skip("P3 actor introspection", repr(e)[:80]); ray.kill(a); return
    res, threads, cid, n = info
    check("[VALID] P3 the actor is assigned at most 1 CPU and no GPU", res.get("CPU", 0) <= 1 and res.get("GPU", 0) == 0, str(res))
    check("[VALID] P3 the actor uses 1 torch thread (no CPU oversubscription)", threads == 1)
    cl = Client(7, SUB[7]["images"], SUB[7]["labels"], seed=0)
    check("[VALID] P3 the actor holds exactly its own client's data", cid == 7 and n == cl.n_train)
    ray.kill(a)

### P4
def determinism_tests():
    section("P4 DETERMINISM AND FAILURE RECOVERY")
    a, b = par_server(), par_server()
    a.train(3, E, LR, BS, m=4); b.train(3, E, LR, BS, m=4)
    check("[VALID] P4 two parallel runs with the same seed give the same weights (completion order does not matter)",
          max_diff(a.model.state_dict(), b.model.state_dict()) < 1e-6)
    p = par_server()
    before = {k: v.clone() for k, v in p.model.state_dict().items()}
    try:
        p.train(1, 1, LR, 0, m=3); raised = False  # batch_size=0 -> DataLoader error inside every actor
    except Exception:
        raised = True
    check("[EDGE] P4 an error inside the actors is raised to the driver", raised)
    check("[EDGE] P4 a failed round leaves the global model untouched (rounds are atomic)", max_diff(before, p.model.state_dict()) == 0.0)
    check("[EDGE] P4 the failed round's actors are still killed (finally block)", wait_no_actors())
    h = p.train(1, 1, LR, BS, m=3)
    check("[EDGE] P4 training continues normally after the failure", np.isfinite(h[0]["train_loss"]))

### P5
def speed_tests():
    section("P5 DOES PARALLELISM PAY OFF?")
    cpus = min(int(ray.cluster_resources().get("CPU", 0)), os.cpu_count() or 1)
    if cpus < 4:
        skip("P5 speed-up", f"only {cpus} usable CPUs"); return
    g = MLP().state_dict()
    _, cl = seq_server()
    EE = 15
    t = time.perf_counter()
    for i in range(4):
        cl[i].local_update(g, EE, LR, BS)
    serial = time.perf_counter() - t
    wall = {}
    for k in (1, 2, 4):
        acts = [ClientActor.remote(i, SUB[i]["images"], SUB[i]["labels"], 0.2, 0) for i in range(k)]
        ray.get([x.local_update.remote(g, 1, LR, BS) for x in acts])  # warm-up: exclude startup
        t = time.perf_counter()
        ray.get([x.local_update.remote(g, EE, LR, BS) for x in acts])
        wall[k] = time.perf_counter() - t
        for x in acts:
            ray.kill(x)
    sp = serial / wall[4]
    print(f"\tserial 4 clients: {serial:.2f}s | actors wall time: {', '.join(f'{k}: {v:.2f}s' for k, v in wall.items())}")
    check("[VALID] P5 4 concurrent actors are at least 1.5x faster than 4 clients in a loop (startup excluded)", sp > 1.5, f"speed-up = {sp:.2f}x")
    check("[VALID] P5 weak scaling: 4 actors take less than 2x the time of 1 actor", wall[4] < 2 * wall[1], f"{wall[4] / wall[1]:.2f}x")

    print("\n\tper-round time, parallel (actor startup included) vs sequential, m=4")
    rows = []
    for e in (1, 5, 10, 20):
        p = par_server(); s, _ = seq_server()
        tp = np.mean([h["time"] for h in p.train(2, e, LR, BS, m=4)])
        t = time.perf_counter(); s.train(2, e, LR, BS, m=4); ts = (time.perf_counter() - t) / 2
        rows.append((e, tp, ts)); print(f"\tE={e:>2}: parallel {tp:5.2f}s | sequential {ts:5.2f}s")
    es, tps = zip(*[(r[0], r[1]) for r in rows])
    slope, icpt = np.polyfit(es, tps, 1)
    print(f"\tparallel round time ~ {icpt:.2f}s + {slope:.3f}s x E  (the intercept is actor startup + serialization)")
    check("[VALID] P5 per-round overhead is positive and visible (intercept > 0), as the assignment's actor-per-round rule implies", icpt > 0)
    check("[VALID] P5 at E=20 the parallel round is faster than the sequential one", rows[-1][1] < rows[-1][2])

def main():
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        print("Ray resources:", ray.cluster_resources())
        equivalence_tests()
        lifecycle_tests()
        resource_tests()
        determinism_tests()
        speed_tests()
    finally:
        ray.shutdown()
    finish()


if __name__ == "__main__":
    main()