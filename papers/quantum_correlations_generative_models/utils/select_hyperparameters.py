"""Apply the Phase 5 tuning rule to the tuning sweep and write the finals sweep spec.

Rule (LOG.md, Phase 5 Comparison Plan and its amendment): per arm and model,
the hyperparameter value with the lowest mean validation NLL at the
best-validation checkpoint over all its tuning runs (h in {4, 8, 12} x seeds
0-2; one deterministic run for k-grams, per order k); values within 1e-3 bits
of that minimum are tied and go to the paper value (lr: the dataset's paper
alpha; pseudo-count 0; smoothing 0.5), otherwise to the tied value closest to
it in log scale (the smaller on equal distance, and the smallest when the
paper value is 0). A value is ineligible when any of its runs FAILED, or
(basis-enhanced kinds) had a unitarity error above 1e-6 at any epoch, i.e.
left the unitary group where its likelihood is no longer normalised
(LOG.md, Phase 5 amendments).

Outputs ``<out>.json`` (every candidate's score, eligibility and the
selection) and the finals spec (arm B with the selected values; arm A
controls B2 with the arm-B lr, B4 at the paper alpha, B3 at the planned
parameter-matched sizes, B5 with the arm-B smoothing).

Usage::

    python3 utils/select_hyperparameters.py <TUNING_SWEEP_DIR> \
        --out results/phase5_tuning_selection --finals-spec configs/sweeps/phase5_finals.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TIE_TOLERANCE_BITS = 1e-3
UNITARITY_TOLERANCE = 1e-6
PAPER_LR = {"biofam": 1e-3, "promoter_gene_sequences": 1e-2, "spect_heart": 1e-2}
PAPER_DEFAULT = {"pseudo_count": 0.0, "smoothing": 0.5}
TUNED_KEY = {
    "basis_enhanced": "lr",
    "basis_enhanced_real": "lr",
    "hmm_gradient": "lr",
    "hmm_baum_welch": "pseudo_count",
    "kgram_counting": "smoothing",
}
BOND_DIMS = list(range(1, 13))
SEEDS = list(range(10))
# Planned B3 sizes (LOG.md, Phase 5 Comparison Plan, as amended after the
# pre-handoff verification: BE count (M + 1) h^2 - 2): arm B uses the
# BE-matched sizes above 12, arm A the BE- and MerLin-matched sizes above 12.
B3_ARM_B = {
    "biofam": [15, 18, 21, 24, 27, 30, 33],
    "promoter_gene_sequences": [15, 17, 19, 21, 24, 26],
    "spect_heart": [14, 16, 17, 19, 21],
}
B3_ARM_A = {
    "biofam": [13, 14, 15, 16, 17, 18, 21, 24, 27, 30, 33],
    "promoter_gene_sequences": [13, 14, 15, 17, 19, 21, 24, 26],
    "spect_heart": [13, 14, 16, 17, 19, 21],
}


def _tie_break(tied: list[float], paper_value: float) -> float:
    if paper_value in tied:
        return paper_value
    if paper_value == 0:
        return min(tied)
    # Rounded so that values equally far in log scale (e.g. 0.001 and 0.1
    # around 0.01) tie exactly and the smaller one wins.
    return min(
        tied, key=lambda v: (round(abs(math.log(v) - math.log(paper_value)), 9), v)
    )


def select(tuning_dir: Path) -> dict:
    """Return every candidate's score and the selected value per (arm, model[, order])."""
    status = json.loads((tuning_dir / "sweep_status.json").read_text())
    summary = json.loads((tuning_dir / "summary.json").read_text())
    failed: set[tuple] = set()
    unstable: dict[tuple, list] = {}
    for run in status["runs"]:
        candidate = run["sweep_candidate"]
        key = (
            candidate["dataset"],
            candidate["config_variant"],
            candidate["model"],
            json.dumps(candidate["hparams"], sort_keys=True),
        )
        if run["status"] != "COMPLETED":
            failed.add(key)
            continue
        if not candidate["model"].startswith("basis_enhanced"):
            continue
        metrics = json.loads((tuning_dir / run["metrics_path"]).read_text())
        drift = max(entry.get("unitarity_error", 0.0) for entry in metrics["history"])
        if drift > UNITARITY_TOLERANCE:
            unstable.setdefault(key, []).append(
                {
                    "bond_dim": candidate["bond_dim"],
                    "seed": run["seed"],
                    "max_unitarity_error": drift,
                }
            )
    groups: dict[tuple, dict[float, dict]] = {}
    for cell in summary["cells"]:
        model = cell["model"]
        hparams = cell["hparams"]
        key_name = TUNED_KEY[model]
        group = (cell["dataset"], cell["config_variant"], model, hparams.get("order"))
        value = float(hparams[key_name])
        entry = groups.setdefault(group, {}).setdefault(
            value, {"weighted_sum": 0.0, "runs": 0, "planned_runs": 0}
        )
        entry["planned_runs"] += cell["expected_runs"]
        if cell["successful_runs"]:
            entry["weighted_sum"] += (
                cell["mean_validation_nll_bits_per_site"] * cell["successful_runs"]
            )
            entry["runs"] += cell["successful_runs"]
        hparam_key = json.dumps(hparams, sort_keys=True)
        candidate_key = (cell["dataset"], cell["config_variant"], model, hparam_key)
        if candidate_key in failed:
            entry["has_failed_run"] = True
        if candidate_key in unstable:
            # ``unstable`` is keyed per candidate (all h); keep only this cell's h
            # so that each unstable run is listed once.
            entry.setdefault("unstable_runs", []).extend(
                run
                for run in unstable[candidate_key]
                if run["bond_dim"] == cell["bond_dim"]
            )
    selections = []
    for (dataset, variant, model, order), candidates in sorted(
        groups.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        key_name = TUNED_KEY[model]
        paper_value = PAPER_LR[dataset] if key_name == "lr" else PAPER_DEFAULT[key_name]
        table = []
        for value, entry in sorted(candidates.items()):
            eligible = (
                not entry.get("has_failed_run")
                and not entry.get("unstable_runs")
                and entry["runs"] == entry["planned_runs"]
            )
            score = entry["weighted_sum"] / entry["runs"] if entry["runs"] else math.inf
            table.append(
                {
                    "value": value,
                    "mean_validation_nll_bits_per_site": score
                    if eligible
                    else math.inf,
                    "observed_mean_validation_nll_bits_per_site": score,
                    "runs": entry["runs"],
                    "planned_runs": entry["planned_runs"],
                    "eligible": eligible,
                    "has_failed_run": bool(entry.get("has_failed_run")),
                    "unstable_runs": entry.get("unstable_runs", []),
                }
            )
        lowest = min(row["mean_validation_nll_bits_per_site"] for row in table)
        tied = [
            row["value"]
            for row in table
            if row["mean_validation_nll_bits_per_site"] - lowest <= TIE_TOLERANCE_BITS
        ]
        selections.append(
            {
                "dataset": dataset,
                "config_variant": variant,
                "arm": dataset + ("_pooled" if "_pooled" in variant else ""),
                "model": model,
                "order": order,
                "hyperparameter": key_name,
                "paper_value": paper_value,
                "candidates": table,
                "tied_values": tied,
                "selected": _tie_break(tied, paper_value),
            }
        )
    return {
        "tuning_sweep": status["sweep_id"],
        "rule": __doc__.split("Outputs")[0].strip(),
        "selections": selections,
        "generated_by": "utils/select_hyperparameters.py",
    }


def finals_spec(selection: dict) -> dict:
    """Return the finals sweep spec built from the selections and the frozen plan."""
    chosen = {
        (s["arm"], s["model"], s["order"]): s["selected"]
        for s in selection["selections"]
    }
    blocks = []
    for arm_variant, dataset, pooled in [
        ("biofam", "biofam", False),
        ("promoter_gene_sequences", "promoter_gene_sequences", False),
        ("spect_heart", "spect_heart", False),
        ("spect_heart_pooled", "spect_heart", True),
    ]:
        variant_b = "_pooled_val" if pooled else "_val"
        variant_a = "_pooled" if pooled else ""
        for model in ("basis_enhanced", "basis_enhanced_real", "hmm_gradient"):
            blocks.append(
                {"datasets": [dataset], "config_variant": variant_b, "model_kinds": [model],
                 "bond_dims": BOND_DIMS, "seeds": SEEDS,
                 "grid": {"lr": [chosen[(arm_variant, model, None)]]}}
            )  # fmt: skip
        blocks.append(
            {"datasets": [dataset], "config_variant": variant_b, "model_kinds": ["hmm_baum_welch"],
             "bond_dims": BOND_DIMS + B3_ARM_B[dataset], "seeds": SEEDS,
             "grid": {"pseudo_count": [chosen[(arm_variant, "hmm_baum_welch", None)]]}}
        )  # fmt: skip
        for order in (1, 2, 3):
            smoothing = chosen[(arm_variant, "kgram_counting", order)]
            for variant in (variant_b, variant_a):
                blocks.append(
                    {"datasets": [dataset], "config_variant": variant, "model_kinds": ["kgram_counting"],
                     "bond_dims": [0], "seeds": [0], "grid": {"order": [order], "smoothing": [smoothing]}}
                )  # fmt: skip
        blocks.append(
            {"datasets": [dataset], "config_variant": variant_a, "model_kinds": ["hmm_gradient"],
             "bond_dims": BOND_DIMS, "seeds": SEEDS,
             "grid": {"lr": [chosen[(arm_variant, "hmm_gradient", None)]]}}
        )  # fmt: skip
        blocks.append(
            {"datasets": [dataset], "config_variant": variant_a, "model_kinds": ["basis_enhanced_real"],
             "bond_dims": BOND_DIMS, "seeds": SEEDS, "grid": {}}
        )  # fmt: skip
        blocks.append(
            {"datasets": [dataset], "config_variant": variant_a, "model_kinds": ["hmm_baum_welch"],
             "bond_dims": B3_ARM_A[dataset], "seeds": SEEDS, "grid": {}}
        )  # fmt: skip
    return {
        "description": "Phase 5 finals (LOG.md, Phase 5 Comparison Plan): arm B (validation-tuned, best-validation checkpoint) "
        "BE, B4, B2, B1 + B3, B5 with the values selected by utils/select_hyperparameters.py from "
        f"{selection['tuning_sweep']}; arm A (paper protocol) B2 at the arm-B lr, B4 at the paper alpha, "
        "B3 at the parameter-matched sizes, B5 at the arm-B smoothing.",
        "blocks": blocks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tuning_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--finals-spec", type=Path, required=True)
    args = parser.parse_args(argv)
    selection = select(args.tuning_dir.resolve())
    args.out.with_suffix(".json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    args.finals_spec.write_text(
        json.dumps(finals_spec(selection), indent=2) + "\n", encoding="utf-8"
    )
    for s in selection["selections"]:
        order = f" k={s['order']}" if s["order"] else ""
        print(
            f"{s['arm']:24s} {s['model']:20s}{order:5s} {s['hyperparameter']}={s['selected']:g} tied={s['tied_values']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
