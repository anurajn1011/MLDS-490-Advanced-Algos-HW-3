# Test Dictionary: Federated Learning and Differential Privacy

Every check in the scripts is tagged in its printed name:

- **`[EDGE]`**: an edge case or limit case. It feeds the code an unusual or extreme input (tiny client, zero epochs, invalid m, crashed actor, b = 0 or 1e6) and checks that it behaves correctly.
- **`[VALID]`**: a validation. It tests a property that must hold if the algorithm is correct (an exact identity, a statistical law, a control experiment, a leakage probe).

Each script ends with a PASS count per tag. A script exits nonzero if any check fails. Run from `src/validation/` (`QUICK=1` shortens the experiments). The letters in the IDs below index the checks within a family in the order they appear in the file.

| Script | Part | Command |
|---|---|---|
| `validation_fedavg.py` | 1.1 sequential FedAvg | `python validation_fedavg.py` |
| `validation_parallel.py` | 1.2 Ray actors | `python validation_parallel.py` |
| `validation_dp.py` | 2 Laplace noise | `python validation_dp.py` |
| `update_agreement.py` | shared non-standard plot | `python update_agreement.py` |

---

# Part 1.1: `validation_fedavg.py`

## Edge-case tests `[EDGE]`

| ID | Edge case | Pass if | Failure suggests |
|---|---|---|---|
| F0e | White-background data: mean pixel after `pre_process` | mean < 0.3 | images not inverted; large-mean inputs blow up SGD (seen at lr = 0.1) |
| F1f | Client with n = 3 (empty validation set, batch size > n) | no crash, `n_val = 0`, finite weights | division by zero or empty-loader crash |
| F2f | All clients the same size | result equals the plain mean | weights not uniform |
| F3a-c | C = 0.02, 0.05, 0.10 | 2, 5, 10 distinct ids | float rounding (`int(0.29*100) = 28`), sampling with replacement |
| F3g-h | m = 0 and m = 101 | exception raised | silent bad values |
| F4c | epochs = 0 | weights identical to the global weights | training loop runs anyway |
| F4d | lr = 0 with E = 4 | weights identical to the global weights | optimizer ignores lr |

## Validation tests `[VALID]`

| ID | Test | Validates | Pass if |
|---|---|---|---|
| F0a | Labels of train and test | valid class indices for 62 outputs | all in [0, 61] |
| F0b | Images vs labels length per client | pairing preserved | equal lengths, 100 clients |
| F0c | MD5-hash train and test images, intersect | no test leakage | overlap = 0 |
| F0d | `pre_process` on an all-white batch | background maps to 0, flattened | shape (3, 784), max 0 |
| F1a | Marker images, n = 118, 331, 393 | split is a partition | disjoint, union = all samples |
| F1b | Same | split fraction | `len(val) == int(0.2 n)` |
| F1c | Same, labels `index % 62` | shuffle keeps labels with images | labels equal `index % 62` |
| F1d | Same seed twice | reproducibility | identical validation set |
| F1e | Other seed / other client id | splits differ | validation sets differ |
| F2a | States 1 and 5 with n = 100 and 300 | sample-count weighting | result = 4 |
| F2b | Average of identical states | weights sum to 1 | output equals input |
| F2c | Random states vs float64 reference | numerical correctness | `allclose`, atol 1e-6 |
| F2d | Clients listed in another order | order invariance | max diff < 1e-6 |
| F2e | Inputs before/after | no in-place mutation | tensors unchanged |
| F3d | 3000 rounds of selection counts | uniform sampling | max z-score < 4.5 |
| F3e | Pick counts vs client size | not size-weighted | abs(corr) < 0.35 |
| F3f | Same seed twice, another once | reproducibility | same, same, different |
| F4a | Global weights before/after `local_update` | input not modified | tensors identical |
| F4b | Returned `n` and metric counts | bookkeeping for exact weighting | counts match, correct <= n |
| F4e | 5 epochs at lr = 0.1 | optimization direction | train loss decreases |
| F4f | 200 local epochs on one client | model and loss can fit data | train accuracy > 0.9 |
| F4g | Scramble validation labels | validation never influences training | weights identical |
| F5a | One round, all clients, E = 1, full batch vs pooled GD step | FedAvg = centralized gradient step | max weight diff < 1e-5 |
| F5b | Same for E = 2, 5, 10 | client drift grows with E | gap(1) < gap(2) < gap(5) < gap(10), gap(10) > 100 x gap(1) |
| F6a | 400 random subsets for m = 2 ... 50 | variance ~ (1/m)(1 - (m-1)/(N-1)) | normalized ratio max/min < 2.5 |
| F6b | Same | more clients, less variance | variance strictly decreasing in m |
| F7a | IID redistribution of the same data; mean cosine between client updates | heterogeneity lowers agreement | IID cosine > non-IID cosine |
| F7b | Validation accuracy, IID vs real | heterogeneity does not help | non-IID <= IID + 0.02 |
| F8a | Test loss of untrained model | near-uniform init | 3.9 < loss < 4.6 (ln 62 = 4.13) |
| F8b | Global model's clean-test accuracy, IID vs real (local validation accuracy is printed but not compared: it flatters non-IID data) | learns beyond majority class | > 3 x majority-class rate |
| F8c | First 10 vs last 10 rounds | loss decreases | last < first |
| F8d | All logged losses | numerical stability | all finite |
| F8e | Shuffle labels within each client | accuracy cannot come from leakage | val acc <= majority rate + 0.05 |
| F9a | NaN-poison `X_test`, train 3 rounds | training never reads test data | losses and weights finite |
| F9b | `evaluate_test` on the poisoned set | the poison is real | returns NaN |
| F9c | Same seed twice | reproducibility | weights identical |
| F9d | Different seed | seed matters | weights differ |

Also saves `figures/val_fedavg_drift_vs_E.png` (from F5).

---

# Part 1.2: `validation_parallel.py`

Uses the first 20 clients. Needs `ray` and `psutil`.

## Edge-case tests `[EDGE]`

| ID | Edge case | Pass if | Failure suggests |
|---|---|---|---|
| P1f | Fewer CPUs than m = 4 (an actor keeps its CPU while alive, so the round would hang, not queue) | cluster CPUs >= 4 | deadlock risk on small machines |
| P1g | m = 1 and m = min(8, CPUs), one round each | matches sequential, max diff < 1e-4 | result depends on actor count |
| P4b | Error inside every actor (`batch_size = 0`) | exception reaches the driver | errors swallowed |
| P4c | Global weights before/after the failed round | unchanged | partial update applied; rounds not atomic |
| P4d | Actor count after the failure | 0 | `finally` / `ray.kill` missing |
| P4e | One normal round after the failure | finite loss | corrupted state |
| P5* | Fewer than 4 usable CPUs | speed tests reported as `SKIP` | speed claims cannot be made |

## Validation tests `[VALID]`

| ID | Test | Validates | Pass if |
|---|---|---|---|
| P1a | Server `n_train` table vs real clients | stub sizes match the split | all equal |
| P1b | Fields of the server's client stubs | server never holds client data | fields = {id, n_train} |
| P1c | Same seed, m = 4, sequential vs parallel | same clients sampled | identical id lists per round |
| P1d | Weights after the rounds | same optimization | max diff < 1e-4 (not bit-identical: actors use 1 thread) |
| P1e | Logged loss and accuracy | same metrics | within 1e-3 |
| P2a | Sampler counts `ClientActor` processes | no 100-actor pool | peak <= 4 |
| P2b | Same | clients really overlap | peak >= 2 |
| P2c | Count after training | `ray.kill` works | 0 within 10 s |
| P3a | Assigned resources inside an actor | cap of 1 CPU, 1 GPU | CPU <= 1, GPU = 0 |
| P3b | `torch.get_num_threads()` in the actor | no oversubscription | 1 |
| P3c | Actor's client id and `n_train` | actor holds only its own data | matches client 7 |
| P4a | Two parallel runs, same seed | completion order does not matter | max diff < 1e-6 |
| P5a | 4 clients in a loop vs 4 warmed-up actors, E = 15 | real speed-up, startup excluded | > 1.5x |
| P5b | Wall time with 1, 2, 4 actors | weak scaling | wall(4) < 2 x wall(1) |
| P5c | Fit round time = intercept + slope x E | startup/serialization overhead exists | intercept > 0 |
| P5d | Parallel vs sequential at E = 20 | parallelism pays off when work dominates | parallel faster |

---

# Part 2: `validation_dp.py`

## Edge-case tests `[EDGE]`

| ID | Edge case | Pass if | Failure suggests |
|---|---|---|---|
| D3b | b = 0 | pixels identical to the clean `Client` | noise path has side effects |
| D3c | Full FedAvg run with b = 0 | history identical to Part 1 | DP code changes the baseline |
| D4d | Clipping to [0, 1] | pixels in [0, 1], smaller distortion than raw noise | clip not applied |
| D4e | b < 0 | `ValueError` | silent misuse |
| D4f | b = 1e-12 | max change < 1e-6 | discontinuity at zero |
| D4g | b = 1e6 | finite float32, same shape | overflow, inf or NaN |
| D6* | Too few exact-0 / exact-1 pixels (< 5000) | attack test reported as `SKIP` | cannot measure attack accuracy |

## Validation tests `[VALID]`

| ID | Test | Validates | Pass if |
|---|---|---|---|
| D0a | Sample mean of the noise (b = 0.5, real `NoisyClient` output) | zero-centred | within 5 standard errors of 0 |
| D0b | Mean absolute value / b | scale parameter | within 1% of 1 |
| D0c | Variance / (2 b^2) | Laplace variance | within 3% of 1 |
| D0d | Excess kurtosis | heavy tails | 3 +- 0.3 (Gaussian: 0) |
| D0e | KS distance to Laplace CDF | distributional fit | D < 0.0036 |
| D0f | KS distance to same-variance Gaussian | test can tell them apart | D_gauss > 5 x D_laplace |
| D1a | Noise vs clean pixel | independent of signal | abs(corr) < 0.01 |
| D1b | Horizontal neighbours | independent per pixel | abs(corr) < 0.01 |
| D1c | Two clients | separate RNG streams | abs(corr) < 0.01 |
| D1d | Two images of one client | per-image noise | abs(corr) < 0.15 |
| D1e | Noise prefixes across images | no repeated pattern | all unique |
| D2a | Two `local_update` calls | perturbed dataset is fixed | `X_train` identical |
| D2b | Rebuild with same / other seed | reproducible, seed-dependent | same, different |
| D2c | Noise at b = 2 vs b = 0.5 | common random numbers across b | ratio exactly 4 (atol 2e-5) |
| D2d | Labels and split vs clean client | noise changes values, not membership | identical |
| D2e | Mean change of validation images | validation also perturbed | about b |
| D3a | Server `X_test` after training on noisy clients | test never perturbed | equals clean preprocessed test |
| D4a-c | Mean norm squared, b = 0.1, 0.5, 1.0 | E[eps^2] = 2 b^2 per pixel (basis of lr scaling) | = clean + 2 x 784 x b^2 within 2% |
| D5a | Nearest-centroid accuracy on perturbed validation, b = 0 ... 5 | more noise never makes data easier | non-increasing (tolerance 0.01) |
| D5b | b = 5 vs b = 0 | noise destroys information | drop >= 0.1 |
| D6a | Attacker thresholds each pixel at 0.5 | b means what we claim | within 0.005 of 1 - 0.5 exp(-1/(2b)) |
| D6b | Same | no attacker beats the eps = 1/b DP bound | <= e^eps/(1+e^eps) + 0.002 |
| D6c | Same | privacy grows with b | non-increasing, last < 0.6 |
| D7a | Part 1 learning rate at every b | the Part 1 lr is stable under noise | all losses finite |
| D7b | b = 0.5, 2: lr grid {0.02, 0.05, 0.1} picked on validation accuracy | the lr need not change with b | lr = 0.05 within 0.02 test accuracy of the validation-picked lr |
| D7c | b = 0.5: old `--scale-lr` (shrinks lr by the ‖x‖² ratio) vs fixed lr | the ‖x‖² scaling rule is wrong (documented finding) | scaled test accuracy < fixed |
| D8a | Mean test accuracy over seeds at fixed lr, b = 0 ... 2 | monotone degradation | non-increasing (tolerance 0.02) |
| D8b | b = 2 vs b = 0 | effect size | drop >= 0.1 |
| D8c | Every b vs b = 0 | noise is not a free gain | acc <= noiseless + 0.02 |

Also saves `figures/val_dp_privacy_utility.png` (attacker accuracy vs b; test accuracy vs attacker accuracy).

---

# Non-standard plot: `update_agreement.py`

Saves `figures/update_agreement.png`. It is a visualization, not a pass/fail test.

| Element | What it shows | How to read it |
|---|---|---|
| Heatmap per setting (Part 1.1, m = 4 parallel setting, Part 2 with b = 0.5) | x = cosine between two clients' updates, y = round, colour = share of client pairs (5-round window) | mass near +1: clients agree; near 0: updates unrelated |
| Coherence curve | \|sum p_k delta_k\| / sum p_k \|delta_k\| per round | 1 = full agreement; dotted line 1/sqrt(m) = unrelated updates |

The m = 4 panel uses sequential clients, justified by P1c-P1d.