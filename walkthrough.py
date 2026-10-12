"""
Draw every figure in the README from the saved results and weights.

    python walkthrough.py        # about 1 minute
"""
from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

from semisup.augment import weak, strong
from semisup.model import SmallCNN
from semisup.fixmatch import predict_proba, pick_labelled
from semisup.phantom import make_slice, make_dataset, SUBTLE

ROOT = Path(__file__).parent
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
EXAMPLE_SEED = 6

# ---- one visual language (matches the other repos in the series) -----------------------
SURFACE, INK, INK2, MUTED, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
GOOD, BAD = "#0ca30c", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRIDC, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
})
METHODS = [("supervised", "labelled scans only", GRAY),
           ("fixmatch", "FixMatch, standard recipe", BLUE),
           ("fixmatch_adapted", "FixMatch, adapted (gentle aug + alignment)", ORANGE)]
NAME = {m: n for m, n, _ in METHODS}
COL = {m: c for m, _, c in METHODS}


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRIDC)


def gray(ax, img, vmin=0, vmax=1):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    clean(ax)


def title(ax, t, sub=None, color=INK):
    ax.set_title(t, fontsize=10.5, fontweight="bold", pad=4, color=color)
    if sub:
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=INK2)


def tidy(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  saved assets/{name}")


def load(name):
    m = SmallCNN()
    m.load_state_dict(torch.load(ROOT / "weights" / name))
    return m.eval()


T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)

# ---- results ------------------------------------------------------------------------------
rows = list(csv.DictReader(open(ROOT / "results" / "label_efficiency.csv")))
budgets = sorted({int(r["n_labels"]) for r in rows})
stats = {}
for m, *_ in METHODS:
    acc = np.array([[float(r["test_accuracy"]) for r in rows if r["method"] == m and int(r["n_labels"]) == n]
                    for n in budgets])
    stats[m] = (acc.mean(1), acc.min(1), acc.max(1))
dyn = np.load(ROOT / "results" / "dynamics_8_labels.npz")
probe_true = dyn["probe_true"].astype(bool)
test_true = dyn["test_true"].astype(bool)


def efficiency_plot(ax, small=False):
    for m, name, col in METHODS:
        mu, lo, hi = stats[m]
        ax.fill_between(budgets, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(budgets, mu, color=col, lw=2.4, marker="o", ms=4 if small else 5, label=name, zorder=3)
    ax.axhline(0.5, color=GRIDC, lw=1, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(0.45, 1.01)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["50%", "60", "70", "80", "90", "100%"])
    tidy(ax)
    if not small:
        ax.set_xlabel("number of labelled scans (plus 10,000 unlabelled for FixMatch)")
        ax.set_ylabel("accuracy on 400 fresh test scans")
        ax.text(budgets[-1], 0.492, "50% = guessing", fontsize=8.5, color=MUTED, va="top", ha="right")


# ---- shared example material ------------------------------------------------------------------
img, info = make_slice(96, True, np.random.default_rng(EXAMPLE_SEED), **SUBTLE)
bright, _ = make_slice(96, True, np.random.default_rng(EXAMPLE_SEED))
XL, yL = make_dataset(2000, seed=31, **SUBTLE)
lab_idx = dyn["labelled_idx"]
XU = make_dataset(10000, seed=21, **SUBTLE)[0][:2000]          # the probe scans used in the dynamics run
final = load("fixmatch_adapted_8labels.pt")
print("drawing figures ...")

# ---- HERO ---------------------------------------------------------------------------------------
fig = plt.figure(figsize=(17, 5.4))
H, y0 = 0.54, 0.2
sq = H * 5.4 / 17
stages = [("A few labelled scans", "8 scans, 4 with a lesion", "what a radiologist has time for", sq),
          ("Lots of unlabelled", "10,000 scans, no labels", "cheap: every hospital has these", sq),
          ("Guess on a weak view", "keep only if ≥ 95% sure", "the guess becomes a 'pseudo-label'", sq),
          ("Train a strong view", "must give the same answer", "learns what really matters", sq),
          ("Result", "accuracy vs labels", "unlabelled data helps, once adapted", sq * 1.25)]
gap = (1 - 0.03 - sum(s[3] for s in stages)) / (len(stages) - 1)
x, boxes = 0.015, []
for i, (t, sub, expl, w) in enumerate(stages):
    boxes.append((x, w))
    fig.text(x + w / 2, y0 + H + 0.15, f"{i + 1}  {t}", ha="center", fontsize=12.5, fontweight="bold")
    fig.text(x + w / 2, y0 + H + 0.085, sub, ha="center", fontsize=10, color=BLUE, family="DejaVu Sans Mono")
    fig.text(x + w / 2, y0 - 0.085, expl, ha="center", fontsize=9.5, color=INK2)
    x += w + gap
xb, wb = boxes[0]
cw, ch = wb / 4 - 0.003, H / 2 - 0.01
for k, i in enumerate(lab_idx[:8]):
    a = fig.add_axes([xb + (k % 4) * (cw + 0.004), y0 + H / 2 + 0.005 - (k // 4) * (ch + 0.02), cw, ch])
    gray(a, XL[i])
    for sp in a.spines.values():
        sp.set_color(ORANGE if yL[i] else BLUE); sp.set_linewidth(2)
xb, wb = boxes[1]
for k in range(16):
    a = fig.add_axes([xb + (k % 4) * wb / 4, y0 + H - (k // 4 + 1) * H / 4, wb / 4 - 0.002, H / 4 - 0.006])
    gray(a, XU[k])
g = torch.Generator().manual_seed(4)
ex_i = int(np.where(probe_true)[0][3])
wv = weak(T(XU[ex_i][None]), g)
sv = strong(T(XU[ex_i][None]), g, gentle=True)
p_w = predict_proba(final, wv)[0, 1].item()
a = fig.add_axes([boxes[2][0], y0, boxes[2][1], H]); gray(a, wv[0].numpy())
a.text(0.5, 0.04, f"P(lesion) = {p_w:.0%}  →  'lesion'", transform=a.transAxes, ha="center", fontsize=10,
       color="white", fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc=ORANGE, ec="none"))
a = fig.add_axes([boxes[3][0], y0, boxes[3][1], H]); gray(a, sv[0].numpy())
a.text(0.5, 0.04, "target: 'lesion'", transform=a.transAxes, ha="center", fontsize=10, color="white",
       fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc=ORANGE, ec="none"))
a = fig.add_axes([boxes[4][0] + 0.03, y0 + 0.03, boxes[4][1] - 0.03, H - 0.03]); efficiency_plot(a, small=True)
a.set_title("orange: adapted · blue: standard · gray: labels only", fontsize=7.5, color=INK2, pad=3)
for (xa, wa), (xc, _) in zip(boxes[:-1], boxes[1:]):
    fig.patches.append(FancyArrowPatch((xa + wa + 0.006, y0 + H / 2), (xc - 0.006, y0 + H / 2),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14,
                                       color=MUTED, lw=1.4))
save(fig, "hero.png")

# ---- 01 · the data ---------------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 0.12, 1.6], hspace=0.1)
sub = gs[:, 0].subgridspec(2, 2, wspace=0.06, hspace=0.25)
for k, (im, t) in enumerate([(bright, "bright (repos 1–3)"), (img, "subtle (this repo)")]):
    a = fig.add_subplot(sub[0, k]); gray(a, im); title(a, t)
    a.add_patch(Circle((info["col"], info["row"]), info["radius"] + 5, fill=False, ec=ORANGE, lw=1.4, ls=(0, (3, 2))))
    z = fig.add_subplot(sub[1, k])
    r0, c0 = int(info["row"]) - 14, int(info["col"]) - 14
    gray(z, im[r0:r0 + 28, c0:c0 + 28]); title(z, "zoomed")
fig.text(0.24, 0.0, "A CNN finds bright lesions with 16 labels (98%), leaving nothing for\nunlabelled data to add. "
         "Faint lesions are harder, and closer to the real problem.", ha="center", va="top", fontsize=9.5,
         color=INK2, linespacing=1.5)
sub2 = gs[:, 2].subgridspec(4, 8, wspace=0.05, hspace=0.08)
for k in range(8):
    i = lab_idx[k]
    a = fig.add_subplot(sub2[0, k]); gray(a, XL[i])
    for sp in a.spines.values():
        sp.set_color(ORANGE if yL[i] else BLUE); sp.set_linewidth(2.2)
    a.set_title("lesion" if yL[i] else "normal", fontsize=8.5, color=ORANGE if yL[i] else BLUE, pad=2,
                fontweight="bold")
for k in range(24):
    a = fig.add_subplot(sub2[1 + k // 8, k % 8]); gray(a, XU[8 + k])
    a.text(0.5, 0.5, "?", transform=a.transAxes, ha="center", va="center", color="white", fontsize=11, alpha=0.6)
fig.text(0.66, 0.93, "8 labelled scans (top) and a few of the 10,000 unlabelled ones", ha="center",
         fontsize=11, fontweight="bold")
fig.text(0.66, 0.0, "Half the unlabelled scans have a faint lesion. Nobody told the model which.",
         ha="center", va="top", fontsize=9.5, color=INK2)
save(fig, "01_the_data.png")

# ---- 02 · weak vs strong augmentation ---------------------------------------------------------------
twin, _ = make_slice(96, False, np.random.default_rng(EXAMPLE_SEED), **SUBTLE)
fig, axs = plt.subplots(3, 5, figsize=(14, 8.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
rowsets = [("weak", lambda x, gg: weak(x, gg), BLUE), ("strong · standard (heavy)", lambda x, gg: strong(x, gg), BLUE),
           ("strong · gentle", lambda x, gg: strong(x, gg, gentle=True), ORANGE)]
survive = {}
for r, (lab, f, col) in enumerate(rowsets):
    for c in range(4):
        gg = torch.Generator().manual_seed(100 + c)
        gray(axs[r, c], f(T(img[None]), gg)[0].numpy())
    axs[r, 0].set_ylabel(lab, fontsize=10.5, fontweight="bold", labelpad=8)
    # how much of the lesion's extra brightness survives? same random draws for the scan and its lesion-free twin
    sig = []
    for k in range(300):
        g1, g2 = torch.Generator().manual_seed(k), torch.Generator().manual_seed(k)
        d = (f(T(img[None]), g1) - f(T(twin[None]), g2))[0].numpy()
        sig.append(d.max() / (img - twin).max())
    survive[lab] = np.array(sig)
a = axs[0, 4]; a.axis("off")
a.text(0.0, 0.5, "Weak views only flip and\nshift: the diagnosis\nobviously stays the same.",
       fontsize=10, va="center", color=INK2, linespacing=1.5)
for r, lab in [(1, "strong · standard (heavy)"), (2, "strong · gentle")]:
    a = axs[r, 4]
    a.hist(survive[lab], bins=np.linspace(0, 1.6, 33), color=ORANGE if r == 2 else BLUE, edgecolor=SURFACE)
    a.axvline(1, color=INK2, lw=1, ls=(0, (3, 2)))
    a.set_title(f"lesion brightness kept · median {np.median(survive[lab]):.0%}", fontsize=9.5, fontweight="bold")
    if r == 2:
        a.set_xlabel("fraction of the lesion's extra brightness left")
    tidy(a); a.set_yticks([])
fig.text(0.5, 0.0, "Heavy intensity changes (gamma, contrast, blur) work well on everyday photos, but here they can wash "
         "a faint lesion into the tissue,\nand heavy noise buries what's left. Then the strong view no longer shows "
         "what its pseudo-label says, and the model learns the wrong lesson.", ha="center", va="top",
         fontsize=9.5, color=INK2, linespacing=1.6)
fig.subplots_adjust(wspace=0.08, hspace=0.32)
save(fig, "02_augmentations.png")

# ---- 03 · one FixMatch step on four unlabelled scans --------------------------------------------------
gw = torch.Generator().manual_seed(7)
Xw = weak(T(XU), gw)
pw = predict_proba(final, Xw)[:, 1].numpy()
conf = np.maximum(pw, 1 - pw) >= 0.95
pick = []
for want in [(True, True, True), (True, False, True), (False, None, None), (True, False, False)]:
    is_conf, guess_lesion, correct = want
    for i in range(len(XU)):
        if i in pick or conf[i] != is_conf:
            continue
        if is_conf and ((pw[i] > 0.5) != guess_lesion or ((pw[i] > 0.5) == probe_true[i]) != correct):
            continue
        pick.append(i)
        break
gs_ = torch.Generator().manual_seed(8)
Xs = strong(T(XU[pick]), gs_, gentle=True)
ps = predict_proba(final, Xs)[:, 1].numpy()
fig, axs = plt.subplots(len(pick), 5, figsize=(14, 3.1 * len(pick)),
                        gridspec_kw={"width_ratios": [1, 1.1, 1, 1.1, 1.3]})
heads = ["1 · weak view", "2 · model's guess", "3 · strong view", "4 · strong view must match", "what really happened"]
for c, h in enumerate(heads):
    axs[0, c].set_title(h, fontsize=10.5, fontweight="bold", pad=8)
for r, i in enumerate(pick):
    gray(axs[r, 0], Xw[i].numpy())
    a = axs[r, 1]
    a.barh([0], [pw[i]], color=ORANGE, height=0.5); a.barh([0], [1], color="none", edgecolor=GRIDC, height=0.5)
    a.axvline(0.95, color=INK2, lw=1, ls=(0, (3, 2))); a.axvline(0.05, color=INK2, lw=1, ls=(0, (3, 2)))
    a.set_xlim(0, 1); a.set_ylim(-1, 1); a.set_yticks([]); a.set_xticks([0, 0.5, 1])
    a.set_xticklabels(["normal", "", "lesion"])
    for sp in a.spines.values():
        sp.set_visible(False)
    a.text(min(max(pw[i], 0.2), 0.8), 0.45, f"P(lesion) {pw[i]:.0%}", ha="center", va="bottom", fontsize=9.5,
           fontweight="bold")
    gray(axs[r, 2], Xs[r].numpy())
    a = axs[r, 3]; a.axis("off")
    if conf[i]:
        lab = "lesion" if pw[i] > 0.5 else "normal"
        a.text(0.5, 0.62, f"pseudo-label: {lab}", ha="center", fontsize=11, fontweight="bold",
               color=ORANGE if lab == "lesion" else BLUE, transform=a.transAxes)
        a.text(0.5, 0.38, f"model now says {ps[r]:.0%} lesion\n→ nudge it toward '{lab}'", ha="center",
               fontsize=9.5, color=INK2, transform=a.transAxes, linespacing=1.5)
    else:
        a.text(0.5, 0.5, "not sure enough\n→ ignored this step", ha="center", va="center", fontsize=11,
               color=MUTED, fontweight="bold", transform=a.transAxes, linespacing=1.5)
    a = axs[r, 4]; a.axis("off")
    truth = "lesion" if probe_true[i] else "normal"
    if conf[i]:
        ok = (pw[i] > 0.5) == probe_true[i]
        a.text(0.5, 0.55, f"truth: {truth}", ha="center", fontsize=11, fontweight="bold", transform=a.transAxes)
        a.text(0.5, 0.33, "✓ helpful pseudo-label" if ok else "✗ WRONG: a faint lesion called\nnormal, and the model learns it",
               ha="center", va="top", fontsize=9.5, color=GOOD if ok else BAD, fontweight="bold",
               transform=a.transAxes, linespacing=1.4)
    else:
        a.text(0.5, 0.5, f"truth: {truth}", ha="center", va="center", fontsize=11, fontweight="bold", transform=a.transAxes)
n_wrong = int((conf & ((pw > 0.5) != probe_true)).sum())
fig.text(0.5, 0.0, f"Adapted FixMatch at the end of training, 8 labelled scans. Of 2,000 unlabelled scans it is confident "
         f"about {conf.mean():.0%};\n{n_wrong} of those confident guesses are still wrong. The truth column is hidden "
         "from the model; it's shown here only for you.", ha="center", va="top", fontsize=9.5, color=INK2,
         linespacing=1.6)
fig.subplots_adjust(wspace=0.15, hspace=0.25)
save(fig, "03_one_fixmatch_step.png")

# ---- 04 · training dynamics ----------------------------------------------------------------------------
steps_p, steps_t = dyn["probe_step"], dyn["test_step"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for m, name, col in METHODS:
    acc = ((dyn[f"{m}_test_p"] > 0.5) == test_true[None]).mean(1)
    axs[0].plot(steps_t, acc, color=col, lw=2.2, label=name)
axs[0].set_title("test accuracy during training", fontsize=10.5, fontweight="bold", loc="left")
axs[0].set_ylim(0.45, 1.0); axs[0].set_xlabel("training step"); tidy(axs[0])
axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
for m in ["fixmatch", "fixmatch_adapted"]:
    P = dyn[f"{m}_probe_p"]
    confident = np.maximum(P, 1 - P) >= 0.95
    called_normal = (P < 0.05).mean(1)
    correct = np.array([((P[k][confident[k]] > 0.5) == probe_true[confident[k]]).mean() if confident[k].any() else np.nan
                        for k in range(len(P))])
    axs[1].plot(steps_p, called_normal, color=COL[m], lw=2.2, label=NAME[m])
    axs[2].plot(steps_p, correct, color=COL[m], lw=2.2)
axs[1].axhline(0.5, color=INK2, lw=1, ls=(0, (3, 2)))
axs[1].text(steps_p[-1], 0.51, "true share of normal scans: 50%", ha="right", va="bottom", fontsize=8.5, color=INK2)
axs[1].set_title("unlabelled scans confidently called 'normal'", fontsize=10.5, fontweight="bold", loc="left")
axs[2].set_title("how many confident guesses are right", fontsize=10.5, fontweight="bold", loc="left")
for a in axs[1:]:
    a.set_xlabel("training step"); tidy(a)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
axs[1].set_ylim(0, 1); axs[2].set_ylim(0.5, 1.0)
fig.text(0.5, -0.06, "8 labelled scans, same random draw for all three. Both FixMatch versions call too many unlabelled "
         "scans 'normal' (they miss the faintest lesions),\nbut the standard one drifts further and, above all, only ~75% of "
         "its confident guesses are right, vs ~91% for the adapted one.\nIt then trains on those mistakes: confirmation "
         "bias. True labels of the unlabelled scans were used only to draw these plots.",
         ha="center", va="top", fontsize=9.5, color=INK2, linespacing=1.6)
save(fig, "04_training_dynamics.png")

# ---- 05 · animation: unlabelled scans separating --------------------------------------------------------
print("  rendering animation ...")
frames = []
bins = np.linspace(0, 1, 41)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
for k in range(len(steps_p)):
    for a, m in zip(axs, ["fixmatch", "fixmatch_adapted"]):
        a.clear()
        P = dyn[f"{m}_probe_p"][k]
        a.hist(P[~probe_true], bins=bins, color=BLUE, alpha=0.85, label="really normal", edgecolor=SURFACE, lw=0.4)
        a.hist(P[probe_true], bins=bins, color=ORANGE, alpha=0.85, label="really a lesion", edgecolor=SURFACE, lw=0.4)
        a.axvspan(0, 0.05, color=BLUE, alpha=0.06); a.axvspan(0.95, 1, color=ORANGE, alpha=0.06)
        a.set_ylim(0, 1000); a.set_xlim(0, 1)
        a.set_xlabel("model's P(lesion) for each unlabelled scan")
        a.set_title(f"{NAME[m]}", fontsize=10.5, fontweight="bold")
        tidy(a)
    axs[0].set_ylabel("number of scans (of 2,000)")
    axs[1].legend(frameon=False, loc="upper center", fontsize=9)
    fig.suptitle(f"training step {steps_p[k]}", fontsize=11, color=INK2, y=0.99)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
plt.close(fig)
small = [f.resize((f.width * 2 // 3, f.height * 2 // 3), Image.LANCZOS).quantize(64) for f in frames]
small[0].save(OUT / "05_unlabelled_scans_separating.gif", save_all=True, append_images=small[1:],
              duration=[220] * (len(small) - 1) + [2500], loop=0, optimize=True)
print("  saved assets/05_unlabelled_scans_separating.gif")

# ---- 06 · label efficiency -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.6))
efficiency_plot(ax)
for m, name, col in METHODS:
    nudge = {"supervised": -6, "fixmatch": 6, "fixmatch_adapted": 0}[m]
    ax.annotate(name, (budgets[-1], stats[m][0][-1]), xytext=(10, nudge), textcoords="offset points",
                va="center", fontsize=9.5, color=INK)
ax.set_title("How much do 10,000 unlabelled scans help?", fontsize=12, fontweight="bold", loc="left", pad=24)
ax.text(0, 1.0, "lines = mean of 3 random draws of labelled scans, shaded = min to max",
        transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
save(fig, "06_label_efficiency.png")
print("done")
