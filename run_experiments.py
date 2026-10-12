"""
Every experiment behind the README, in one script.

    python run_experiments.py         # about 1 hour on a laptop CPU

1. Label efficiency, 4 to 32 labelled scans, 3 random draws each, three methods:
     supervised         labelled scans only
     fixmatch           the standard recipe (heavy strong augmentation)
     fixmatch_adapted   gentle strong augmentation + distribution alignment
2. Training dynamics: one run of each FixMatch variant with 8 labels, recording its guesses
   on 2,000 unlabelled scans (their true labels are used ONLY to draw the figures).

Hyperparameters follow the FixMatch paper (threshold 0.95, unlabelled-loss weight 1, SGD
lr 0.03 with cosine decay). One change for CPU: 4 unlabelled scans per labelled one (paper: 7).

The adapted recipe was developed while looking at the usual test set (seed 2), so all
numbers here are reported on a FRESH test set (seed 51) that was never used before.
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch

from semisup.fixmatch import train, accuracy, pick_labelled, predict_proba
from semisup.phantom import make_dataset, SUBTLE

ROOT = Path(__file__).parent
BUDGETS = [4, 8, 16, 32]
METHODS = {"supervised": None, "fixmatch": dict(), "fixmatch_adapted": dict(gentle=True, align=True)}
SEEDS = [0, 1, 2]
STEPS = 1500


def pseudo_label_quality(model, X, y, threshold=0.95):
    """How many unlabelled scans the final model is confident about, and how many of those are right."""
    p = predict_proba(model, X)
    conf, guess = p.max(1)
    keep = conf >= threshold
    acc = (guess[keep] == y[keep]).float().mean().item() if keep.any() else float("nan")
    return keep.float().mean().item(), acc


def main():
    t0 = time.time()
    print("generating data (subtle lesions) ...")
    XU, yU_hidden = make_dataset(10000, seed=21, **SUBTLE)   # unlabelled; labels kept ONLY for analysis
    XL, yL = make_dataset(2000, seed=31, **SUBTLE)           # pool we draw the few labelled scans from
    XT, yT = make_dataset(400, seed=51, **SUBTLE)            # fresh test set, never used during development
    XU, XL, XT = (torch.tensor(a, dtype=torch.float32) for a in (XU, XL, XT))
    tU, tL, tT = (torch.tensor(a) for a in (yU_hidden, yL, yT))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "weights").mkdir(exist_ok=True)

    # 1 · label efficiency -------------------------------------------------------------------
    rows = []
    for n in BUDGETS:
        t = time.time()
        for s in SEEDS:
            idx = pick_labelled(yL, n, s)
            for method, kw in METHODS.items():
                if kw is None:
                    m, _ = train(XL[idx], tL[idx], None, steps=STEPS, seed=s)
                    rows.append((n, method, s, accuracy(m, XT, tT), "", ""))
                else:
                    m, _ = train(XL[idx], tL[idx], XU, steps=STEPS, seed=s, **kw)
                    cov, pacc = pseudo_label_quality(m, XU[:2000], tU[:2000])
                    rows.append((n, method, s, accuracy(m, XT, tT), f"{cov:.4f}", f"{pacc:.4f}"))
        print(f"  {n:3d} labels  " + "  ".join(
            f"{k} {np.mean([r[3] for r in rows if r[0] == n and r[1] == k]):.1%}" for k in METHODS)
            + f"  ({time.time() - t:.0f}s)", flush=True)
    with open(ROOT / "results" / "label_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_labels", "method", "seed", "test_accuracy", "pseudo_label_coverage", "pseudo_label_accuracy"])
        w.writerows(rows)

    # 2 · training dynamics (8 labels, seed 0) --------------------------------------------------
    print("training-dynamics runs ...")
    idx = pick_labelled(yL, 8, 0)
    probe = XU[:2000]
    out = dict(labelled_idx=idx, probe_true=yU_hidden[:2000], test_true=yT)
    for method, kw in METHODS.items():
        Xu = None if kw is None else XU
        m, lg = train(XL[idx], tL[idx], Xu, steps=STEPS, seed=0, X_probe=probe, probe_every=50,
                      X_test=XT, test_every=50, **(kw or {}))
        out[f"{method}_probe_p"] = np.array(lg["probe_p"], dtype=np.float32)
        out[f"{method}_test_p"] = np.array(lg["test_p"], dtype=np.float32)
        out[f"{method}_mask_rate"] = np.array(lg["mask_rate"])
        out["probe_step"], out["test_step"], out["log_step"] = lg["probe_step"], lg["test_step"], lg["step"]
        torch.save(m.state_dict(), ROOT / "weights" / f"{method}_8labels.pt")
    np.savez_compressed(ROOT / "results" / "dynamics_8_labels.npz", **out)

    print(f"done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
