"""Plot the Phase 4 MerLin adaptation against the Phase 3 models, per arm and h.

Reads ``results/photonic_comparison.json`` (``utils/compare_photonic.py``,
built from the sweeps' structured artifacts). Rows: train and test mean over
10 trials with 1-SEM error bars in the paper metric NLL - log2(N)/n, then the
log2 herald success per sequence of the MerLin runs (mean, with the min-max
range over seeds). Columns: biofam, Promoters, SPECT standard split, SPECT
pooled split. Cells flagged by the pre-declared "beats the converged HMM" rule
are ringed.

The figure is written to ``--out-dir`` and to every passed sweep directory.

Usage::

    python3 utils/plot_photonic_comparison.py results/photonic_comparison.json \
        --sweep-dirs <MERLIN_SWEEP_DIR> [...] --out-dir results \
        --name photonic_merlin_comparison
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
    "spect_heart_pooled": "SPECT Heart\n(pooled 267-row split, paper-inferred)",
}
# Reference categorical slots 1-3 (dataviz default palette; the first three
# slots validate all-pairs), with distinct markers so identity is not
# colour-alone. Slots 1-2 keep the Phase 3 figure assignment.
MODEL_STYLE = {
    "hmm": {"color": "#2a78d6", "marker": "o", "label": "HMM (Baum-Welch), Phase 3"},
    "basis_enhanced": {"color": "#eb6834", "marker": "x", "label": "Basis-enhanced 2-gram, Phase 3"},
    "merlin": {"color": "#1baf7a", "marker": "s", "label": "MerLin adaptation (simulated, 2 photons)"},
}  # fmt: skip
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _style_axis(axis) -> None:
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.set_xticks(range(2, 13, 2))


def plot(comparison: dict, out_paths: list[Path]) -> None:
    rows = comparison["rows"]
    arms = [arm for arm in ARM_TITLES if any(r["arm"] == arm for r in rows)]
    figure, axes = plt.subplots(
        3, len(arms), figsize=(4.2 * len(arms), 9.6), squeeze=False
    )
    for column, arm in enumerate(arms):
        arm_rows = sorted(
            (r for r in rows if r["arm"] == arm), key=lambda r: r["bond_dim"]
        )
        bond_dims = [r["bond_dim"] for r in arm_rows]
        for row_index, split in enumerate(("train", "test")):
            axis = axes[row_index][column]
            for model, style in MODEL_STYLE.items():
                axis.errorbar(
                    bond_dims, [r[f"{model}_mean_{split}"] for r in arm_rows],
                    yerr=[r[f"{model}_sem_{split}"] for r in arm_rows],
                    color=style["color"], marker=style["marker"], markersize=5,
                    linewidth=1.5, linestyle="--", capsize=2, label=style["label"],
                )  # fmt: skip
            flagged = [r for r in arm_rows if r[f"beats_converged_hmm_{split}"]]
            if flagged:
                axis.plot(
                    [r["bond_dim"] for r in flagged], [r[f"merlin_mean_{split}"] for r in flagged],
                    linestyle="none", marker="o", markersize=13, markerfacecolor="none",
                    markeredgecolor="#0b0b0b", markeredgewidth=1.2,
                    label="MerLin beats converged HMM (pre-declared 2-SEM rule)",
                )  # fmt: skip
            _style_axis(axis)
            if row_index == 0:
                axis.set_title(ARM_TITLES[arm], fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{split.capitalize()}: NLL - log2(N)/n (bits/site)")
        axis = axes[2][column]
        means = [r["herald_success_log2_mean"] for r in arm_rows]
        axis.fill_between(
            bond_dims, [r["herald_success_log2_min"] for r in arm_rows],
            [r["herald_success_log2_max"] for r in arm_rows],
            color=MODEL_STYLE["merlin"]["color"], alpha=0.18, linewidth=0,
            label="MerLin herald success, min-max over 10 seeds",
        )  # fmt: skip
        axis.plot(
            bond_dims, means, color=MODEL_STYLE["merlin"]["color"], marker="s",
            markersize=5, linewidth=1.5, label="MerLin herald success, mean over 10 seeds",
        )  # fmt: skip
        _style_axis(axis)
        axis.set_xlabel("Bond dimension h (hidden modes)")
        if column == 0:
            axis.set_ylabel("log2 P(all n heralds succeed)")
    handles, labels = [], []
    for axis in (axes[0][0], *[axes[0][c] for c in range(len(arms))], axes[2][0]):
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(
        handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9
    )
    figure.suptitle(
        "Mean of 10 trials; error bars 1 SEM. Scope: MerLin adaptation (simulated, exact, "
        "shots = 0), full h = 1..12 grid, 10 seeds; not the paper's model",
        fontsize=10, color=TEXT_SECONDARY,
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0.07, 1, 0.97))
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
    out_paths = [args.out_dir / f"{args.name}.png"] + [
        d.resolve() / f"{args.name}.png" for d in args.sweep_dirs
    ]
    plot(json.loads(args.comparison.read_text()), out_paths)
    print("\n".join(str(path) for path in out_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
