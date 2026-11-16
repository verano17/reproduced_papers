"""Plot the HMM-convergence diagnostic: paper HMM vs Baum-Welch stopped at K iterations.

Reads the JSON written by ``utils/diagnose_hmm_iterations.py`` and the sweep
summaries, and draws per dataset the best-of-10 training value (paper metric)
of the HMM after K = 10, 50, 100 and the final (<= 2000) Baum-Welch
iterations, the basis-enhanced model, and the paper's plotted HMM and
basis-enhanced points. A dataset present in several sweeps is taken from the
last one passed.

Usage::

    python3 utils/plot_hmm_caps.py <SWEEP_DIR> [<SWEEP_DIR> ...] \
        --caps-json results/hmm_iteration_caps.json \
        --paper-values results/paper_values_fig7_fig8.json --out-dir results \
        --name hmm_convergence_diagnostic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC = "paper_kl_estimate_bits_per_site"
HMM_BLUE_STEPS = {10: "#9ec5f4", 50: "#5598e7", 100: "#2a78d6", 2000: "#104281"}
BE_ORANGE = "#eb6834"
TITLES = {
    "biofam": "Biofam",
    "promoter_gene_sequences": "Promoter Gene Sequences",
    "spect_heart": "SPECT Heart",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument("--caps-json", type=Path, required=True)
    parser.add_argument("--paper-values", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    sweep_dirs = [path.resolve() for path in args.sweep_dirs]
    caps = json.loads(args.caps_json.read_text())
    paper = json.loads(args.paper_values.read_text())["fig7_best_of_ten"]

    hmm_rows: dict = {}
    be_cells: dict = {}
    source: dict = {}
    for sweep_dir in sweep_dirs:
        summary = json.loads((sweep_dir / "summary.json").read_text())
        sweep_caps = next(
            s for s in caps["sweeps"] if s["sweep_id"] == summary["sweep_id"]
        )
        for dataset in {cell["dataset"] for cell in summary["cells"]}:
            source[dataset] = summary["sweep_id"]
            hmm_rows[dataset] = sorted(
                (r for r in sweep_caps["cells"] if r["dataset"] == dataset),
                key=lambda r: r["bond_dim"],
            )
            be_cells[dataset] = sorted(
                (
                    c
                    for c in summary["cells"]
                    if c["dataset"] == dataset and c["model"] == "basis_enhanced"
                ),
                key=lambda c: c["bond_dim"],
            )

    datasets = [d for d in TITLES if d in source]
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 4.4))
    for axis, dataset in zip(axes, datasets):
        rows = hmm_rows[dataset]
        bond_dims = [r["bond_dim"] for r in rows]
        for cap, color in HMM_BLUE_STEPS.items():
            label = (
                f"HMM, final (≤ {cap} it.)"
                if cap == 2000
                else f"HMM, {cap} EM iterations"
            )
            axis.plot(
                bond_dims, [r[f"train_at_{cap}"] for r in rows], color=color, marker="o",
                markersize=4, linewidth=1.5, label=label,
            )  # fmt: skip
        axis.plot(
            [c["bond_dim"] for c in be_cells[dataset]],
            [c[f"best_train_train_{METRIC}"] for c in be_cells[dataset]],
            color=BE_ORANGE, marker="x", markersize=5, linewidth=1.5, linestyle="--",
            label="Basis-enhanced, 150/75 epochs",
        )  # fmt: skip
        for model, marker, color in (
            ("hmm_baum_welch", "o", "#2a78d6"),
            ("basis_enhanced", "s", BE_ORANGE),
        ):
            values = paper[dataset]["train"][model]
            axis.plot(
                [int(h) for h in values], list(values.values()), linestyle="none",
                marker=marker, markersize=9, markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.5,
                label=f"paper {'HMM' if model == 'hmm_baum_welch' else 'basis-enhanced'} (plot-read)",
            )  # fmt: skip
        title = TITLES[dataset]
        if dataset == "spect_heart":
            title += (
                " (pooled split)"
                if "pooled" in source[dataset]
                else " (standard split)"
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Bond dimension h")
        axis.set_xticks(range(2, 13, 2))
        axis.set_xlim(1.5, 12.5)
        axis.grid(color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Train NLL − log2(N)/n, best of 10 (bits/site)")
    for axis, dataset in zip(axes, datasets):
        low = min(
            min(r["train_at_2000"] for r in hmm_rows[dataset] if r["bond_dim"] > 1),
            min(
                c[f"best_train_train_{METRIC}"]
                for c in be_cells[dataset]
                if c["bond_dim"] > 1
            ),
        )
        high = max(r["train_at_10"] for r in hmm_rows[dataset] if r["bond_dim"] > 1)
        margin = 0.1 * (high - low)
        axis.set_ylim(low - margin, high + margin)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8
    )
    figure.suptitle(
        "Diagnostic: HMM baseline stopped early vs converged (no new runs; read from per-iteration histories). Scope: paper-accurate, 10 seeds",
        fontsize=10, color="#52514e",
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0.1, 1, 0.95))
    out_paths = [args.out_dir / f"{args.name}.png"] + [
        d / f"{args.name}.png" for d in sweep_dirs
    ]
    for path in out_paths:
        figure.savefig(path, dpi=150)
    plt.close(figure)
    print("\n".join(str(path) for path in out_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
