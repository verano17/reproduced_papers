"""Plot the Fig. 8 analogue: training gap vs the 5-sigma likelihood-ratio threshold.

Reads ``summary.json`` of every passed sweep (a dataset present in several
sweeps is taken from the last one passed) and draws, for Promoters and SPECT:
the gap n x (best HMM train NLL - best basis-enhanced train NLL) in bits per
sequence (dashed, this work), the 5-sigma Wilks threshold with the
paper-inferred df = h^2 - 1 (solid; LOG.md R7), and the paper's gap and
threshold read from Fig. 8 (open markers and dotted line). Above the solid
line the null hypothesis "the basis-enhanced optimum reduces to an HMM" is
rejected at 5 sigma.

Usage::

    python3 utils/plot_fig8.py <SWEEP_DIR> [<SWEEP_DIR> ...] \
        --paper-values results/paper_values_fig7_fig8.json --out-dir results \
        --name fig8_reproduction --scope "paper-accurate, 10 seeds"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATASET_STYLE = {
    "promoter_gene_sequences": {"color": "#2a78d6", "label": "Promoter Gene Sequences"},
    "spect_heart": {"color": "#eb6834", "label": "SPECT Heart"},
}


def load_fig8(sweep_dirs: list[Path]) -> tuple[dict, dict]:
    entries: dict = {}
    source: dict = {}
    for sweep_dir in sweep_dirs:
        summary = json.loads((sweep_dir / "summary.json").read_text())
        if summary["coverage"] != "COMPLETE SWEEP":
            raise ValueError(
                f"{sweep_dir} is {summary['coverage']}; plot complete sweeps only"
            )
        for dataset in {entry["dataset"] for entry in summary["fig8"]}:
            entries[dataset] = sorted(
                (e for e in summary["fig8"] if e["dataset"] == dataset),
                key=lambda e: e["bond_dim"],
            )
            source[dataset] = summary["sweep_id"]
    return entries, source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    parser.add_argument("--paper-values", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args(argv)
    sweep_dirs = [path.resolve() for path in args.sweep_dirs]
    entries, source = load_fig8(sweep_dirs)
    paper = json.loads(args.paper_values.read_text())

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for axis, (dataset, style) in zip(axes, DATASET_STYLE.items()):
        rows = entries[dataset]
        bond_dims = [e["bond_dim"] for e in rows]
        axis.plot(
            bond_dims, [e["gap_bits_per_sequence_nll_bits_per_site"] for e in rows],
            color=style["color"], linestyle="--", marker="o", markersize=5, linewidth=1.5,
            label="gap, this work",
        )  # fmt: skip
        threshold_rows = [
            e for e in rows if e["lrt_5sigma_threshold_paper_df_bits_per_sequence"]
        ]
        axis.plot(
            [e["bond_dim"] for e in threshold_rows],
            [e["lrt_5sigma_threshold_paper_df_bits_per_sequence"] for e in threshold_rows],
            color=style["color"], linewidth=2, label="5σ threshold, df = h²−1, this split",
        )  # fmt: skip
        gap = paper["fig8_gap_bits_per_sequence"][dataset]
        axis.plot(
            [int(h) for h in gap], list(gap.values()), linestyle="none", marker="s",
            markersize=8, markerfacecolor="none", markeredgecolor="#52514e",
            markeredgewidth=1.3, label="gap, paper (plot-read)",
        )  # fmt: skip
        threshold = paper["fig8_threshold_bits_per_sequence"][dataset]
        axis.plot(
            [int(h) for h in threshold], list(threshold.values()), color="#52514e",
            linestyle=":", linewidth=1.5, label="threshold, paper (plot-read)",
        )  # fmt: skip
        axis.axhline(0, color="#a3a29c", linewidth=0.8)
        title = style["label"]
        if dataset == "spect_heart":
            title += (
                " (pooled split)"
                if "pooled" in source[dataset]
                else " (standard split)"
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Bond dimension h")
        axis.set_xticks(range(2, 13, 2))
        axis.grid(color="#e4e3df", linewidth=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("HMM − basis-enhanced train NLL (bits/sequence)")
    figure.suptitle(
        f"Fig. 8 analogue: best-of-10 training gap vs likelihood-ratio threshold. Scope: {args.scope}",
        fontsize=10, color="#52514e",
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0, 1, 0.94))
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
