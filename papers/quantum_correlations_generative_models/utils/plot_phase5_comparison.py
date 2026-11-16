"""Plot the Phase 5 comparison as BE's advantage over each control, per arm and h.

Reads ``results/phase5_comparison.json`` (``utils/compare_phase5.py``). Each
point is mean(control) - mean(BE) in bits/site (positive: the basis-enhanced
model is better); the error bar is the pre-declared separation margin
2 sqrt(SEM_BE^2 + SEM_control^2), so a point is a separation exactly when its
bar does not cross zero (filled marker), and a tie otherwise (hollow marker).

Figure 1 (``<name>.png``): rows arm A train, arm A test, arm B train, arm B
test; columns biofam, Promoters, SPECT standard split, SPECT pooled split;
controls B1, B2, B3 (at the BE-matched h'), B4, and B5 k = 2 (reference).
Figure 2 (``<name>_merlin.png``): the MerLin adaptation (arm A) against B1 at
equal h and B3 at MerLin's matched h'.

Usage::

    python3 utils/plot_phase5_comparison.py results/phase5_comparison.json \
        --sweep-dirs <FINALS_SWEEP_DIR> --out-dir results --name phase5_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_TITLES = {
    "biofam": "Biofam",
    "promoter_gene_sequences": "Promoter Gene Sequences",
    "spect_heart": "SPECT Heart\n(standard split, as stated)",
    "spect_heart_pooled": "SPECT Heart\n(pooled split, paper-inferred)",
}
ROW_TITLES = {
    ("A", "train"): "Arm A (paper protocol)\ntrain",
    ("A", "test"): "Arm A (paper protocol)\ntest",
    ("B", "train"): "Arm B (validation-tuned)\ntrain",
    ("B", "test"): "Arm B (validation-tuned)\ntest",
}
# Reference categorical slots 1-5 in fixed order, each with its own marker so
# identity never relies on colour alone.
CONTROL_STYLE = {
    "B1": {"color": "#2a78d6", "marker": "o", "label": "B1 converged Baum-Welch HMM, equal h"},
    "B2": {"color": "#eb6834", "marker": "s", "label": "B2 gradient-trained HMM, equal h"},
    "B3": {"color": "#1baf7a", "marker": "^", "label": "B3 Baum-Welch HMM, parameter-matched h'"},
    "B4": {"color": "#eda100", "marker": "D", "label": "B4 real-orthogonal BE, equal h"},
    "B5_k2": {"color": "#e87ba4", "marker": "v", "label": "B5 first-order Markov chain (k = 2), reference"},
}  # fmt: skip
MERLIN_STYLE = {
    "B1": {"color": "#2a78d6", "marker": "o", "label": "vs B1 converged Baum-Welch HMM, equal h"},
    "B3_merlin": {"color": "#eb6834", "marker": "s", "label": "vs Baum-Welch HMM matched to MerLin's parameter count"},
}  # fmt: skip
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
ZERO = "#0b0b0b"


def _style_axis(axis) -> None:
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.axhline(0.0, color=ZERO, linewidth=0.8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_xticks(range(2, 13, 2))


def _draw(axis, rows: list[dict], subject: str, controls: dict) -> None:
    offsets = {
        name: (i - (len(controls) - 1) / 2) * 0.12 for i, name in enumerate(controls)
    }
    for name, style in controls.items():
        # Decision range h = 2..12 only (LOG.md, Phase 5 decision rules); h = 1
        # stays in the JSON / CSV but would set the y-scale on its own.
        points = [
            r
            for r in rows
            if r["bond_dim"] >= 2
            and r.get(f"{subject}_minus_{name}_advantage") is not None
        ]
        h = [r["bond_dim"] + offsets[name] for r in points]
        advantage = [r[f"{subject}_minus_{name}_advantage"] for r in points]
        margin = [r[f"{subject}_vs_{name}_margin"] for r in points]
        separated = [abs(a) > m for a, m in zip(advantage, margin, strict=True)]
        axis.errorbar(
            h, advantage, yerr=margin, color=style["color"], linestyle="none",
            elinewidth=1.2, capsize=2, marker="none",
        )  # fmt: skip
        for filled in (True, False):
            chosen = [i for i, s in enumerate(separated) if s == filled]
            axis.plot(
                [h[i] for i in chosen], [advantage[i] for i in chosen],
                linestyle="none", marker=style["marker"], markersize=6,
                markerfacecolor=style["color"] if filled else "white",
                markeredgecolor=style["color"], markeredgewidth=1.2,
                label=style["label"] if filled else None,
            )  # fmt: skip


def _legend(figure, axes, extra: list) -> None:
    handles, labels = [], []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    for handle, label in extra:
        handles.append(handle)
        labels.append(label)
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9
    )


def _marker_key() -> list:
    filled = plt.Line2D(
        [], [], linestyle="none", marker="o", color="#52514e", markersize=6
    )
    hollow = plt.Line2D(
        [], [], linestyle="none", marker="o", markerfacecolor="white",
        markeredgecolor="#52514e", markersize=6,
    )  # fmt: skip
    return [
        (filled, "filled: separated (bar clears 0, pre-declared 2-SEM rule)"),
        (hollow, "hollow: no separation (tie)"),
    ]


def plot_controls(comparison: dict, out_paths: list[Path]) -> None:
    rows = comparison["rows"]
    arms = list(ARM_TITLES)
    figure, axes = plt.subplots(
        4, len(arms), figsize=(4.4 * len(arms), 13.5), squeeze=False
    )
    for row_index, (protocol, split) in enumerate(ROW_TITLES):
        for column, arm in enumerate(arms):
            axis = axes[row_index][column]
            selected = [
                r for r in rows if (r["arm"], r["protocol"], r["split"]) == (arm, protocol, split)
            ]  # fmt: skip
            _draw(axis, selected, "BE", CONTROL_STYLE)
            _style_axis(axis)
            if row_index == 0:
                axis.set_title(ARM_TITLES[arm], fontsize=10)
            if column == 0:
                axis.set_ylabel(
                    f"{ROW_TITLES[(protocol, split)]}\nmean(control) - mean(BE), bits/site",
                    fontsize=9,
                )  # fmt: skip
            if row_index == 3:
                axis.set_xlabel("Bond dimension h of BE")
    _legend(figure, [a for row in axes for a in row], _marker_key())
    figure.suptitle(
        "Basis-enhanced 2-gram (BE) advantage over each control; above 0 = BE better. "
        "Mean of 10 trials, error bar = 2 sqrt(SEM_BE^2 + SEM_control^2),\n" "h = 2..12 (decision range). Local runs; "
        "arm B biofam h = 1-2 missing (failed / invalid BE and B4 runs)",
        fontsize=10, color=TEXT_SECONDARY,
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    for path in out_paths:
        figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_merlin(comparison: dict, out_paths: list[Path]) -> None:
    rows = comparison["rows"]
    arms = list(ARM_TITLES)
    figure, axes = plt.subplots(
        2, len(arms), figsize=(4.4 * len(arms), 7.4), squeeze=False
    )
    for row_index, split in enumerate(("train", "test")):
        for column, arm in enumerate(arms):
            axis = axes[row_index][column]
            selected = [
                r for r in rows if (r["arm"], r["protocol"], r["split"]) == (arm, "A", split)
            ]  # fmt: skip
            _draw(axis, selected, "MerLin", MERLIN_STYLE)
            _style_axis(axis)
            if row_index == 0:
                axis.set_title(ARM_TITLES[arm], fontsize=10)
            if column == 0:
                axis.set_ylabel(
                    f"Arm A {split}\nmean(control) - mean(MerLin), bits/site",
                    fontsize=9,
                )
            if row_index == 1:
                axis.set_xlabel("Bond dimension h (hidden modes)")
    _legend(figure, [a for row in axes for a in row], _marker_key())
    figure.suptitle(
        "MerLin adaptation (simulated, exact, shots = 0; not the paper's model) against "
        "fair classical HMMs, h = 2..12; above 0 = MerLin better. Mean of 10 trials, error bar = 2-SEM margin",
        fontsize=10, color=TEXT_SECONDARY,
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    for path in out_paths:
        figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--sweep-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    comparison = json.loads(args.comparison.read_text())
    directories = [args.out_dir] + [d.resolve() for d in args.sweep_dirs]
    controls = [d / f"{args.name}.png" for d in directories]
    merlin = [d / f"{args.name}_merlin.png" for d in directories]
    plot_controls(comparison, controls)
    plot_merlin(comparison, merlin)
    print("\n".join(str(path) for path in controls + merlin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
