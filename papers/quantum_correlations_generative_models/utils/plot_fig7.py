"""Plot the Fig. 7 (best of 10 trials) or Fig. 13 (mean of 10 trials) analogue.

Reads ``summary.json`` of every passed sweep (``utils/summarize_fig7.py``); a
dataset present in several sweeps is taken from the last one passed, so
``<E5 sweep> <pooled SPECT sweep>`` replaces the SPECT column with the
pooled-split arm. Values are in the paper's metric NLL - log2(N)/n (LOG.md R1).
Paper values read from the figure (``results/paper_values_fig7_fig8.json``) are
overlaid as open markers (Fig. 7 only; Fig. 13 was not digitised). Error bars
are 1 SEM over the 10 trials, as in the paper.

The figure is written to ``--out-dir`` and to every passed sweep directory.

Usage::

    python3 utils/plot_fig7.py <SWEEP_DIR> [<SWEEP_DIR> ...] --statistic best \
        --paper-values results/paper_values_fig7_fig8.json --out-dir results \
        --name fig7_reproduction --scope "paper-accurate, 10 seeds"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC = "paper_kl_estimate_bits_per_site"
DATASET_TITLES = {
    "biofam": "Biofam",
    "promoter_gene_sequences": "Promoter Gene Sequences",
    "spect_heart": "SPECT Heart",
}
# Reference categorical slots 1 and 2 (dataviz default palette), matching the
# paper's colour assignment (HMM blue circles, basis-enhanced orange crosses).
MODEL_STYLE = {
    "hmm_baum_welch": {"color": "#2a78d6", "marker": "o", "label": "HMM (Baum-Welch)"},
    "basis_enhanced": {
        "color": "#eb6834",
        "marker": "x",
        "label": "Basis-enhanced 2-gram",
    },
}
TEXT_SECONDARY = "#52514e"


def load_cells(sweep_dirs: list[Path]) -> tuple[dict, dict]:
    """Return {(dataset, model, h): cell} and {dataset: sweep_id}."""
    cells: dict = {}
    source: dict = {}
    for sweep_dir in sweep_dirs:
        summary = json.loads((sweep_dir / "summary.json").read_text())
        if summary["coverage"] != "COMPLETE SWEEP":
            raise ValueError(
                f"{sweep_dir} is {summary['coverage']}; plot complete sweeps only"
            )
        datasets = {cell["dataset"] for cell in summary["cells"]}
        for dataset in datasets:
            for key in [k for k in cells if k[0] == dataset]:
                del cells[key]
            source[dataset] = summary["sweep_id"]
        for cell in summary["cells"]:
            cells[(cell["dataset"], cell["model"], cell["bond_dim"])] = cell
    return cells, source


def plot(cells, source, paper_values, statistic, scope, out_paths):
    datasets = [d for d in DATASET_TITLES if d in source]
    figure, axes = plt.subplots(
        2, len(datasets), figsize=(4.2 * len(datasets), 6.6), squeeze=False
    )
    for column, dataset in enumerate(datasets):
        for row, split in enumerate(("train", "test")):
            axis = axes[row][column]
            for model, style in MODEL_STYLE.items():
                bond_dims = sorted(
                    h for d, m, h in cells if d == dataset and m == model
                )
                if statistic == "best":
                    values = [
                        cells[(dataset, model, h)][f"best_train_{split}_{METRIC}"]
                        for h in bond_dims
                    ]
                else:
                    values = [
                        cells[(dataset, model, h)][f"mean_{split}_{METRIC}"]
                        for h in bond_dims
                    ]
                errors = [
                    cells[(dataset, model, h)][f"sem_{split}_{METRIC}"]
                    for h in bond_dims
                ]
                axis.errorbar(
                    bond_dims, values, yerr=errors, color=style["color"], marker=style["marker"],
                    markersize=5, linewidth=1.5, linestyle="--", capsize=2, label=f"{style['label']} (this work)",
                )  # fmt: skip
                paper = paper_values["fig7_best_of_ten"][dataset][split][model]
                if statistic == "best":
                    axis.plot(
                        [int(h) for h in paper], list(paper.values()), linestyle="none",
                        marker="o" if model == "hmm_baum_welch" else "s", markersize=9,
                        markerfacecolor="none", markeredgecolor=style["color"], markeredgewidth=1.5,
                        label=f"{style['label']} (paper, plot-read)",
                    )  # fmt: skip
            axis.grid(color="#e4e3df", linewidth=0.6)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if row == 0:
                title = DATASET_TITLES[dataset]
                if "pooled" in source[dataset]:
                    title += "\n(pooled 267-row split, paper-inferred)"
                elif dataset == "spect_heart":
                    title += "\n(standard split, as stated)"
                axis.set_title(title, fontsize=10)
            if row == 1:
                axis.set_xlabel("Bond dimension h")
            if column == 0:
                axis.set_ylabel(f"{split.capitalize()}: NLL - log2(N)/n (bits/site)")
            axis.set_xticks(range(2, 13, 2))
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9
    )
    what = (
        "Best of 10 trials (lowest train NLL)"
        if statistic == "best"
        else "Mean of 10 trials"
    )
    figure.suptitle(
        f"{what}; error bars 1 SEM over 10 trials. Scope: {scope}",
        fontsize=10, color=TEXT_SECONDARY,
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0.08, 1, 0.97))
    for path in out_paths:
        figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument("--statistic", choices=("best", "mean"), required=True)
    parser.add_argument("--paper-values", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args(argv)
    sweep_dirs = [path.resolve() for path in args.sweep_dirs]
    cells, source = load_cells(sweep_dirs)
    out_paths = [args.out_dir / f"{args.name}.png"] + [
        d / f"{args.name}.png" for d in sweep_dirs
    ]
    plot(
        cells,
        source,
        json.loads(args.paper_values.read_text()),
        args.statistic,
        args.scope,
        out_paths,
    )
    print("\n".join(str(path) for path in out_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
