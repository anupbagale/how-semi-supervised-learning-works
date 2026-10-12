"""Checks for the augmentations, the model and the FixMatch loop."""
import numpy as np
import torch

from semisup.augment import weak, strong
from semisup.model import SmallCNN
from semisup.fixmatch import train, pick_labelled, EMA

torch.manual_seed(0)


def test_augmentations_keep_shape_and_range():
    x = torch.rand(4, 96, 96)
    for f in (weak, strong, lambda x: strong(x, gentle=True)):
        y = f(x)
        assert y.shape == x.shape
        assert y.min() >= 0 and y.max() <= 1


def test_weak_is_only_flip_and_shift():
    """A flat image stays flat in the middle under a weak view (no intensity changes)."""
    x = torch.full((2, 96, 96), 0.5)
    y = weak(x)
    assert torch.allclose(y[:, 20:76, 20:76], torch.full((2, 56, 56), 0.5), atol=1e-5)


def test_gentle_changes_intensity_less_than_heavy():
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    x = torch.full((64, 96, 96), 0.45)
    heavy = strong(x, g1)[:, 30:66, 30:66].mean((1, 2))
    gentle = strong(x, g2, gentle=True)[:, 30:66, 30:66].mean((1, 2))
    assert (gentle - 0.45).abs().mean() < (heavy - 0.45).abs().mean()


def test_model_output_and_balanced_pick():
    assert SmallCNN()(torch.rand(5, 96, 96)).shape == (5, 2)
    labels = np.array([0, 1] * 50)
    idx = pick_labelled(labels, 16, seed=0)
    assert len(idx) == 16 and labels[idx].sum() == 8


def test_unlabelled_loss_only_counts_confident_guesses():
    """With an impossible threshold, nothing is pseudo-labelled, so the unlabelled loss is exactly 0."""
    X = torch.rand(8, 96, 96)
    y = torch.tensor([0, 1] * 4)
    _, log = train(X, y, X_unlab=torch.rand(32, 96, 96), steps=30, batch=4, mu=2, threshold=1.01)
    assert max(log["loss_u"]) == 0.0 and max(log["mask_rate"]) == 0.0


def test_ema_moves_toward_the_model():
    m = SmallCNN()
    e = EMA(m, decay=0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    before = next(e.model.parameters()).clone()
    e.update(m)
    after = next(e.model.parameters())
    assert torch.allclose(after - before, torch.full_like(before, 0.5))


def test_distribution_alignment_runs():
    X = torch.rand(8, 96, 96)
    y = torch.tensor([0, 1] * 4)
    model, log = train(X, y, X_unlab=torch.rand(32, 96, 96), steps=10, batch=4, mu=2, align=True, gentle=True)
    assert len(log["mask_rate"]) == 1
