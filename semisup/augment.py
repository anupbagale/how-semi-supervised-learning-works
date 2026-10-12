"""
Two kinds of augmentation, the heart of FixMatch.

weak   : things that obviously don't change the diagnosis
         (left-right flip, a small shift)
strong : much harder distortions of the same scan
         (bigger shift, rotation, zoom, contrast/brightness/gamma changes, blur, noise)

FixMatch's idea: if the model is confident about the WEAK view, that guess
becomes the target for the STRONG view. A model that gives the same answer
under heavy distortion has learned what actually matters.

Medical-imaging catch: the "heavy" intensity changes that work well on everyday
photos can wash out a FAINT lesion, so the strong view no longer shows what the
pseudo-label says. `gentle=True` keeps the geometry but limits intensity changes.
"""
import math

import torch
import torch.nn.functional as F


def _affine(x, max_shift, max_rot_deg, scale_range, generator=None):
    """Random flip + shift + rotation + zoom for a batch (B, H, W), via grid_sample."""
    B, H, W = x.shape
    g = generator
    flip = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0)
    ang = (torch.rand(B, generator=g) * 2 - 1) * math.radians(max_rot_deg)
    sc = scale_range[0] + torch.rand(B, generator=g) * (scale_range[1] - scale_range[0])
    tx = (torch.rand(B, generator=g) * 2 - 1) * max_shift * 2 / W
    ty = (torch.rand(B, generator=g) * 2 - 1) * max_shift * 2 / H
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1),
                         torch.stack([sin * flip, cos, ty], 1)], 1)       # (B, 2, 3)
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    return F.grid_sample(x[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]


def weak(x, generator=None):
    return _affine(x, max_shift=4, max_rot_deg=0, scale_range=(1.0, 1.0), generator=generator)


def strong(x, generator=None, gentle=False):
    """
    gentle=False: the usual heavy recipe (big intensity changes, blur, noise).
    gentle=True : same geometry, but intensity changes small enough not to wash out a faint lesion.
    """
    g = generator
    B = x.shape[0]
    x = _affine(x, max_shift=10, max_rot_deg=20, scale_range=(0.85, 1.15), generator=g)
    r = lambda lo, hi: (lo + torch.rand(B, 1, 1, generator=g) * (hi - lo))
    gam, con, bri, noise = ((0.85, 1.2), (0.85, 1.15), 0.05, 0.03) if gentle else ((0.6, 1.6), (0.6, 1.4), 0.15, 0.08)
    x = x.clamp(0, 1) ** r(*gam)                                         # gamma
    mean = x.mean((1, 2), keepdim=True)
    x = (x - mean) * r(*con) + mean + r(-bri, bri)                       # contrast, brightness
    if not gentle and torch.rand(1, generator=g) < 0.5:                  # blur on half the batches
        k = torch.tensor([1., 2., 1.])
        k = (torch.outer(k, k) / 16)[None, None]
        x = F.conv2d(x[:, None], k, padding=1)[:, 0]
    x = x + torch.randn(x.shape, generator=g) * r(0.0, noise)            # scanner-like noise
    return x.clamp(0, 1)
