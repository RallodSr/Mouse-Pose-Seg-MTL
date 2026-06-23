"""
Training-dynamics figure for HybridMTLNet (Fig. 3): validation/training mIoU
and PCK over 100 epochs, with learning-rate-decay markers and the peak-PCK
point. Reads the seeded 1:500 run log so the curve matches the reported numbers.

Usage:
    python app/plot_training_curves.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path("models/checkpoints/sweep_1_500/train_log.csv")
OUT = Path("paper/figures/training_curves.pdf")


def main():
    rows = list(csv.DictReader(open(LOG)))
    ep = [int(r["epoch"]) for r in rows]
    tr_miou = [float(r["train_miou"]) for r in rows]
    va_miou = [float(r["val_miou"]) for r in rows]
    tr_pck = [float(r["train_pck"]) for r in rows]
    va_pck = [float(r["val_pck"]) for r in rows]

    # learning-rate decay epochs (where lr drops)
    lr = [float(r["lr"]) for r in rows]
    decays = [ep[i] for i in range(1, len(lr)) if lr[i] < lr[i - 1]]
    # peak validation PCK
    pk_i = max(range(len(va_pck)), key=lambda i: va_pck[i])

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(7.2, 2.3))

    c_miou, c_pck = "#1f6fb4", "#d1660f"
    ax.plot(ep, va_miou, color=c_miou, lw=1.8, label="mIoU (val)")
    ax.plot(ep, tr_miou, color=c_miou, lw=1.0, ls="--", alpha=0.7, label="mIoU (train)")
    ax.plot(ep, va_pck, color=c_pck, lw=1.8, label="PCK (val)")
    ax.plot(ep, tr_pck, color=c_pck, lw=1.0, ls="--", alpha=0.7, label="PCK (train)")

    for d in decays:
        ax.axvline(d, color="gray", ls=":", lw=0.9)
        ax.text(d + 0.6, 0.07, f"LR$\\downarrow$ (ep {d})", rotation=90,
                va="bottom", ha="left", fontsize=7, color="gray")

    ax.scatter([ep[pk_i]], [va_pck[pk_i]], color=c_pck, s=22, zorder=5,
               edgecolor="white", linewidth=0.6)
    ax.annotate(f"peak PCK {va_pck[pk_i]:.4f}\n(ep {ep[pk_i]})",
                xy=(ep[pk_i], va_pck[pk_i]), xytext=(ep[pk_i] - 30, va_pck[pk_i] - 0.22),
                fontsize=7, color=c_pck,
                arrowprops=dict(arrowstyle="->", color=c_pck, lw=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_xlim(0, max(ep))
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout(pad=0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"saved -> {OUT}")
    print(f"  LR decays @ epochs {decays}; peak val PCK {va_pck[pk_i]:.4f} @ ep {ep[pk_i]}")


if __name__ == "__main__":
    main()
