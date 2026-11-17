"""
val_common.py - shared helpers for the three validation scripts.

Layout assumed:
    src/q1/fed_avg/ : model.py, clients.py, server.py
    src/q1/parallel_clients/ : paths.py, client_actor.py, parallel_server.py
    src/q2/ : paths.py, noisy_client.py, run_dp.py
    src/validation/ : this folder, every script should be run from this folder

Every check prints PASS / FAIL / SKIP. finish() exits nonzero if any check failed.
Set QUICK=1 for shorter experiments, as a consequence the learning checks are then less reliable.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
TRAIN_PATH = ROOT / "data/Assignment3-data/train_data.npy"
TEST_PATH = ROOT / "data/Assignment3-data/test_data.npy"
FIG = ROOT / "validation_figures"
FIG.mkdir(exist_ok=True)
QUICK = bool(os.environ.get("QUICK"))

def add_paths(*rel):
    for r in rel:
        p = str(SRC / r)
        if p not in sys.path:
            sys.path.insert(0, p)

def load_data():
    return (np.load(TRAIN_PATH, allow_pickle=True),
            np.load(TEST_PATH, allow_pickle=True))

def flat(state):
    """Flatten a state_dict into one float vector."""
    return torch.cat([v.detach().flatten().float() for v in state.values()])

def max_diff(a, b):
    return (flat(a) - flat(b)).abs().max().item()

_results = []
_tags = {}

def section(title):
    print(f"\n=== {title} ===")

def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append(ok)
    tag = "EDGE" if name.startswith("[EDGE]") else "VALID"
    _tags.setdefault(tag, []).append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    return ok

def skip(name, why):
    print(f"[SKIP] {name}   ({why})")

def finish():
    n_fail = _results.count(False)
    for tag, v in sorted(_tags.items()):
        print(f"{tag:<5}: {sum(v)}/{len(v)} passed")
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    sys.exit(1 if n_fail else 0) # Success exit is 0