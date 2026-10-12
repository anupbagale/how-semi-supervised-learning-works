"""
FixMatch (Sohn et al., 2020), written out step by step.

Each training step uses two batches:
  labelled   : a few scans with a known answer       -> ordinary cross-entropy
  unlabelled : many scans with NO answer
      1. look at a WEAK view, predict              p = model(weak(x))
      2. if the model is confident, max(p) >= 0.95, its guess becomes a "pseudo-label"
      3. the model must predict that same label for a STRONG view of the same scan
      4. unconfident scans are simply ignored this step

Supervised-only training is the same loop with the unlabelled part switched off,
so the comparison is fair: same model, optimiser, augmentations and step count.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .augment import weak, strong
from .model import SmallCNN


class EMA:
    """Keep a slowly-moving average of the weights; it is steadier than the raw model."""
    def __init__(self, model, decay=0.99):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                e.copy_(p)


@torch.no_grad()
def predict_proba(model, X, batch=500):
    model.eval()
    return torch.cat([model(X[i:i + batch]).softmax(1) for i in range(0, len(X), batch)])


def train(X_lab, y_lab, X_unlab=None, steps=1500, batch=16, mu=4, threshold=0.95, lambda_u=1.0,
          lr=0.03, seed=0, X_probe=None, probe_every=None, X_test=None, test_every=None,
          snapshot_steps=(), gentle=False, warmup=0, align=False):
    """
    Returns (ema_model, log). With X_unlab=None this is plain supervised training.
    X_probe (optional): unlabelled scans to record the model's guesses on, for the figures.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = SmallCNN()
    ema = EMA(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: math.cos(7 * math.pi * s / (16 * steps)))
    log = {"step": [], "loss_l": [], "loss_u": [], "mask_rate": [], "probe_step": [], "probe_p": [],
           "test_step": [], "test_p": [], "snapshots": {}}
    n_lab = len(X_lab)
    # distribution alignment: the labelled scans tell us the expected class balance
    target_dist = torch.bincount(y_lab, minlength=2).float() / n_lab
    running_p = torch.full((2,), 0.5)

    for step in range(steps):
        model.train()
        li = torch.randint(0, n_lab, (batch,), generator=g)
        xl = weak(X_lab[li], g)
        if X_unlab is None:
            logits_l = model(xl)
            loss_u, mask_rate = torch.zeros(()), 0.0
        else:
            ui = torch.randint(0, len(X_unlab), (mu * batch,), generator=g)
            xu = X_unlab[ui]
            xu_w, xu_s = weak(xu, g), strong(xu, g, gentle=gentle)
            logits = model(torch.cat([xl, xu_w, xu_s]))
            logits_l = logits[:batch]
            logits_w, logits_s = logits[batch:].chunk(2)
            probs = logits_w.detach().softmax(1)
            if align:
                # If the model has drifted towards calling most unlabelled scans "normal",
                # rescale its guesses so on average they match the labelled class balance.
                running_p = 0.99 * running_p + 0.01 * probs.mean(0)        # ~ last 100 batches
                probs = probs * target_dist / running_p
                probs = probs / probs.sum(1, keepdim=True)
            conf, pseudo = probs.max(1)
            mask = (conf >= threshold).float()
            loss_u = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
            mask_rate = mask.mean().item()
        loss_l = F.cross_entropy(logits_l, y_lab[li])
        ramp = min(1.0, step / warmup) if warmup else 1.0     # optionally ease the unlabelled loss in
        loss = loss_l + lambda_u * ramp * loss_u
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ema.update(model)

        if step % 25 == 0:
            log["step"].append(step); log["loss_l"].append(loss_l.item())
            log["loss_u"].append(loss_u.item()); log["mask_rate"].append(mask_rate)
        if X_probe is not None and probe_every and (step % probe_every == 0 or step == steps - 1):
            log["probe_step"].append(step)
            log["probe_p"].append(predict_proba(ema.model, X_probe)[:, 1].numpy())
        if step in snapshot_steps:
            log["snapshots"][step] = copy.deepcopy(ema.model.state_dict())
        if X_test is not None and test_every and (step % test_every == 0 or step == steps - 1):
            log["test_step"].append(step)
            log["test_p"].append(predict_proba(ema.model, X_test)[:, 1].numpy())
    return ema.model, log


def accuracy(model, X, y):
    return (predict_proba(model, X).argmax(1) == y).float().mean().item()


def pick_labelled(labels, n, seed):
    """n labelled scans, half with a lesion and half without."""
    g = np.random.default_rng(seed)
    labels = np.asarray(labels)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    return np.concatenate([g.choice(pos, n // 2, replace=False), g.choice(neg, n // 2, replace=False)])
