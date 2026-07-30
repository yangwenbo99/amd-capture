#!/usr/bin/env python3
"""
plot_alignment_shifts.py  —  Visualise quad-alignment drift across a capture set.

Loads every ``*.align.opt.json`` produced by ``optimize_alignment.py`` (batch
mode writes one per captured image) from a directory and plots how the four
quadrilateral vertices vary from capture to capture.

Because different captures may reference images of different pixel dimensions,
the *relative* vertex coordinates (``x_rel`` / ``y_rel``, both in [0, 1] of the
reference frame) are used so every capture is directly comparable.  Files are
sorted by name, which — given the zero-padded ``cap-NNNN-KKK`` convention —
matches capture order.

Plots (all distributions; capture order is irrelevant)
------------------------------------------------------
1. 2-D scatter of all four corner positions (shows spatial spread).
2. Histogram of per-corner displacement magnitude from the median position.
3. Histogram of per-corner x-shift from the median.
4. Histogram of per-corner y-shift from the median.
5. Histogram of the final optimisation loss (data-quality indicator).

Usage
-----
    python3 plot_alignment_shifts.py <output-dir> \
        [--output shifts.png] [--dpi 150] [--units rel|ref-px|cap-px]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


CORNERS = ("tl", "tr", "br", "bl")
CORNER_LABELS = {"tl": "TL", "tr": "TR", "br": "BR", "bl": "BL"}
CORNER_COLORS = {
    "tl": "tab:blue",
    "tr": "tab:orange",
    "br": "tab:green",
    "bl": "tab:red",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(input_dir: Path) -> list[dict]:
    """Load and name-sort every *.align.opt.json in input_dir."""
    files = sorted(input_dir.glob("*.align.opt.json"))
    if not files:
        print(
            f"error: no *.align.opt.json files found in {input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    records: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skipping {f.name}: {exc}", file=sys.stderr)
            continue
        if not all(k in data.get("vertices", {}) for k in CORNERS):
            print(f"[warn] skipping {f.name}: missing vertex keys", file=sys.stderr)
            continue
        data["_filename"] = f.name
        records.append(data)

    if not records:
        print("error: no usable records loaded", file=sys.stderr)
        sys.exit(1)
    return records


def extract(records: list[dict], units: str):
    """
    Return:
      verts   dict corner -> ndarray (N, 2) of (x, y) in the chosen units
      labels  list[str] of filenames (capture order)
      losses  ndarray (N,) of final optimisation loss (nan if absent)
      unit_lbl human-readable axis-unit string
    """
    n = len(records)
    verts = {c: np.zeros((n, 2)) for c in CORNERS}
    labels: list[str] = []
    losses = np.full(n, np.nan)

    for i, r in enumerate(records):
        vd = r["vertices"]
        ref = r.get("reference", {})
        cap = r.get("captured", {})
        for c in CORNERS:
            if units == "rel":
                x, y = vd[c]["x_rel"], vd[c]["y_rel"]
            elif units == "ref-px":
                x, y = vd[c]["x_abs"], vd[c]["y_abs"]
            elif units == "cap-px":
                # Relative position scaled to captured-image pixels, so the
                # shift is expressed in the frame you actually photographed.
                x = vd[c]["x_rel"] * cap.get("width", 1.0)
                y = vd[c]["y_rel"] * cap.get("height", 1.0)
            else:  # pragma: no cover - guarded by argparse choices
                raise ValueError(units)
            verts[c][i] = (x, y)
        labels.append(r["_filename"])
        losses[i] = r.get("optimization", {}).get("final_loss", np.nan)

    unit_lbl = {
        "rel": "fraction of reference frame",
        "ref-px": "reference-image pixels",
        "cap-px": "captured-image pixels",
    }[units]
    return verts, labels, losses, unit_lbl


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(verts: dict, unit_lbl: str) -> None:
    print(f"\nvertex shift summary ({unit_lbl}):")
    header = f"  {'corner':6s}{'median x':>12s}{'median y':>12s}" \
             f"{'std x':>10s}{'std y':>10s}{'max |disp|':>12s}"
    print(header)
    for c in CORNERS:
        xy = verts[c]
        med = np.median(xy, axis=0)
        std = xy.std(axis=0)
        disp = np.linalg.norm(xy - med, axis=1)
        print(
            f"  {CORNER_LABELS[c]:6s}{med[0]:12.4f}{med[1]:12.4f}"
            f"{std[0]:10.4f}{std[1]:10.4f}{disp.max():12.4f}"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _hist_corners(ax, data_fn, bins, unit_lbl, title, xlabel):
    """Plot one overlapping step-histogram per corner on ax."""
    import matplotlib.pyplot as plt  # noqa: F401 – called in figure context
    for c in CORNERS:
        values = data_fn(c)
        ax.hist(values, bins=bins, histtype="step", linewidth=1.5,
                color=CORNER_COLORS[c], label=CORNER_LABELS[c], density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def make_figure(verts: dict, labels: list[str], losses: np.ndarray, unit_lbl: str,
                title: str):
    import matplotlib.pyplot as plt

    # Pre-compute per-corner shift from median (centred at 0, comparable across corners).
    medians = {c: np.median(verts[c], axis=0) for c in CORNERS}
    shifts  = {c: verts[c] - medians[c] for c in CORNERS}   # (N, 2), zero-centred

    bins = "auto"

    fig = plt.figure(figsize=(15, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    fig.suptitle(title, fontsize=13)

    # ----- Panel 1: 2-D corner scatter -----
    ax = fig.add_subplot(gs[0, 0])
    for c in CORNERS:
        xy  = verts[c]
        med = medians[c]
        ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.45,
                   color=CORNER_COLORS[c], label=CORNER_LABELS[c])
        ax.scatter([med[0]], [med[1]], s=90, marker="x",
                   color=CORNER_COLORS[c], linewidths=2)
    ax.set_title("Corner positions  (x = median)")
    ax.set_xlabel(f"x  ({unit_lbl})")
    ax.set_ylabel(f"y  ({unit_lbl})")
    ax.invert_yaxis()   # image coords: y grows downward
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="center", markerscale=1.5, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ----- Panel 2: histogram of displacement magnitude -----
    ax = fig.add_subplot(gs[0, 1])
    _hist_corners(
        ax,
        data_fn=lambda c: np.linalg.norm(shifts[c], axis=1),
        bins=bins,
        unit_lbl=unit_lbl,
        title="Distribution of displacement magnitude\n(from per-corner median)",
        xlabel=f"|displacement|  ({unit_lbl})",
    )

    # ----- Panel 3: histogram of final loss -----
    ax = fig.add_subplot(gs[0, 2])
    finite = losses[np.isfinite(losses)]
    if finite.size:
        ax.hist(finite, bins=bins, color="steelblue", edgecolor="white",
                linewidth=0.5, density=True)
    else:
        ax.text(0.5, 0.5, "no loss data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Distribution of final optimisation loss")
    ax.set_xlabel("final_loss")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)

    # ----- Panel 4: histogram of x-shift -----
    ax = fig.add_subplot(gs[1, 0:2])
    _hist_corners(
        ax,
        data_fn=lambda c: shifts[c][:, 0],
        bins=bins,
        unit_lbl=unit_lbl,
        title="Distribution of x-shift from per-corner median",
        xlabel=f"x shift  ({unit_lbl})",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # ----- Panel 5: histogram of y-shift -----
    ax = fig.add_subplot(gs[1, 2])
    _hist_corners(
        ax,
        data_fn=lambda c: shifts[c][:, 1],
        bins=bins,
        unit_lbl=unit_lbl,
        title="Distribution of y-shift from per-corner median",
        xlabel=f"y shift  ({unit_lbl})",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot quad-alignment vertex shifts from a directory of "
                    "optimize_alignment.py output JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_dir",
                   help="directory containing *.align.opt.json files")
    p.add_argument("--output", default=None,
                   help="save figure to this path (e.g. shifts.png). "
                        "If omitted, the figure is shown interactively.")
    p.add_argument("--units", choices=("rel", "ref-px", "cap-px"), default="rel",
                   help="coordinate units for the plots: relative to the "
                        "reference frame, reference-image pixels, or "
                        "captured-image pixels")
    p.add_argument("--dpi", type=int, default=150,
                   help="resolution when saving the figure")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"error: not a directory: {input_dir}", file=sys.stderr)
        return 2

    # Use a non-interactive backend when writing straight to a file.
    import matplotlib
    if args.output:
        matplotlib.use("Agg")

    records = load_records(input_dir)
    verts, labels, losses, unit_lbl = extract(records, args.units)
    print(f"loaded {len(records)} alignment file(s) from {input_dir}")

    print_summary(verts, unit_lbl)

    title = f"Alignment shifts — {input_dir.name}  ({len(records)} captures)"
    fig = make_figure(verts, labels, losses, unit_lbl, title)

    if args.output:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
        print(f"saved: {out}")
    else:
        import matplotlib.pyplot as plt
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
