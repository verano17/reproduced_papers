"""Apply the frozen Phase 5 decision rules to every comparison cell, per arm and h.

Sources (all read from structured sweep artifacts, never from logs):

- Phase 3 sweeps: BE and B1 under the paper protocol (arm A);
- Phase 4 MerLin sweeps: the MerLin adaptation (arm A only);
- the Phase 5 finals sweep: arm A controls (B2, B4, B3 above h' = 12, B5)
  and every arm B model (validation-tuned, best-validation checkpoint).

Rules (LOG.md, "Phase 5 Comparison Plan — 2026-11-12T16:24:33Z" and its
amendments), on per-site NLL in bits (lower is better):

- Separation: X beats Y if mean(Y) - mean(X) > 2 sqrt(SEM_X^2 + SEM_Y^2).
- C3 "learns better" (arm A, train) / "generalizes better" (arm B, test):
  the advantage holds at h if BE beats B1, B2 and B3 (B3 at the BE-matched
  h'). SUPPORTED at >= 9 of h = 2..12, PARTIALLY_SUPPORTED at 3-8,
  UNSUPPORTED at <= 2, UNRESOLVED if a required cell is missing.
  biofam "comparable": no baseline separated from BE by more than 0.05
  bits/site at h >= 4 (read as: a separation under the 2-SEM rule whose
  mean difference also exceeds 0.05 bits/site).
- C4: the Phase 3 5-sigma training gap (B1, best-of-10, df = h^2 - 1 as in
  Phase 3) counts at h only where BE best-of-10 train is also below B3
  best-of-10 train.
- C5: the C3 rule on arm A means, train and test.
- Sec. VI: complex amplitudes matter at h if BE beats B4 on train (arm A) and
  on test (arm B); SUPPORTED at >= 3 values of h in 2..12. With missing cells
  the verdict is UNRESOLVED when the missing values could change it.
- MerLin (arm A only): against B1 at equal h and B3 at MerLin's matched h'.
- B5: reference only, against BE at every h, per order k.

Writes ``<out>.json`` (verdicts, commensurability, systems table, per-h rows)
and ``<out>.csv`` (per-h rows).

Usage::

    python3 utils/compare_phase5.py \
        --phase3 outdir/sweeps/e5_fig7_20261112-100722 outdir/sweeps/e5_fig7_spect_pooled_20261112-114050 \
        --merlin outdir/sweeps/e5_merlin_20261112-132728 outdir/sweeps/e5_merlin_spect_pooled_20261112-164320 \
        --finals outdir/sweeps/phase5_finals_20261112-183133 \
        --out results/phase5_comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PAPERS_DIR))
from quantum_correlations_generative_models.lib.basis_enhanced import (  # noqa: E402
    parameter_matched_hmm_bond_dim,
)

METRIC = "nll_bits_per_site"
SEM_MULTIPLIER = 2.0
COMPARABLE_TOLERANCE_BITS = 0.05
BOND_DIMS = list(range(1, 13))
DECISION_BOND_DIMS = list(range(2, 13))
KGRAM_ORDERS = (1, 2, 3)
ARMS = ("biofam", "promoter_gene_sequences", "spect_heart", "spect_heart_pooled")
C3_ARMS = ("promoter_gene_sequences", "spect_heart", "spect_heart_pooled")
ALPHABET = {"biofam": 8, "promoter_gene_sequences": 4, "spect_heart": 2}
MODEL_LABEL = {
    "basis_enhanced": "BE",
    "hmm_baum_welch": "B1",
    "hmm_gradient": "B2",
    "basis_enhanced_real": "B4",
    "kgram_counting": "B5",
    "photonic_merlin": "MerLin",
}
ROLE = {
    "BE": "paper method (basis-enhanced 2-gram)",
    "B1": "paper baseline, converged Baum-Welch HMM at equal h",
    "B2": "gradient-trained HMM at equal h (optimizer control)",
    "B3": "parameter-matched Baum-Welch HMM at h' (capacity control)",
    "B4": "real-orthogonal BE at equal h (complex-amplitude ablation)",
    "B5": "order-k counting model (reference only)",
    "MerLin": "MerLin adaptation (simulated), not the paper's model",
    "B3_merlin": "Baum-Welch HMM matched to MerLin's parameter count",
}
PROTOCOL_SCOPE = {
    "A": "paper protocol: no validation split, final-epoch checkpoint (local)",
    "B": "validation-tuned: 20% validation carve-out, best-validation checkpoint (local)",
}


def _protocol(config_variant: str | None) -> str:
    return "B" if config_variant and config_variant.endswith("_val") else "A"


def _arm(dataset: str, config_variant: str | None) -> str:
    return dataset + (
        "_pooled" if config_variant and "_pooled" in config_variant else ""
    )


def _cell_record(cell: dict, sweep_id: str, protocol: str) -> dict:
    record = {
        "sweep_id": sweep_id,
        "status": cell["status"],
        "successful_runs": cell["successful_runs"],
        "expected_runs": cell["expected_runs"],
        "invalid_runs": cell.get("invalid_runs", []),
        "free_parameter_count": cell.get("free_parameter_count"),
        "wall_seconds_total": cell.get("wall_seconds_total"),
        "hparams": cell.get("hparams", {}),
    }
    if not cell["successful_runs"]:
        return record
    for split in ("train", "test"):
        record[f"mean_{split}"] = cell[f"mean_{split}_{METRIC}"]
        record[f"sem_{split}"] = cell[f"sem_{split}_{METRIC}"]
        best = "best_validation" if protocol == "B" else "best_train"
        record[f"best_of_10_{split}"] = cell[f"{best}_{split}_{METRIC}"]
        for name in (record[f"mean_{split}"], record[f"sem_{split}"]):
            if not math.isfinite(name):
                raise ValueError(f"Non-finite aggregate in {sweep_id} cell {cell}")
    return record


def _legacy_wall_seconds(sweep_dir: Path, status: dict) -> dict[tuple, float]:
    """Sum ``wall_seconds`` of completed runs per (arm, model, h) for Phase 3/4 sweeps."""
    variant = status["plan"].get("config_variant")
    totals: dict[tuple, float] = {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        key = (
            _arm(metrics["dataset"], variant),
            metrics["model_kind"],
            metrics["bond_dim"],
        )
        totals[key] = totals.get(key, 0.0) + metrics["wall_seconds"]
    return totals


def load_cells(
    phase3_dirs: list[Path], merlin_dirs: list[Path], finals_dir: Path
) -> tuple[dict, dict, dict]:
    """Return (cells, fig8 entries, source sweep ids).

    ``cells`` is keyed by (arm, protocol, label, h, order) with order None
    except for B5, whose h is 0.
    """
    cells: dict[tuple, dict] = {}
    fig8: dict[tuple, dict] = {}
    sources: dict[str, list[str]] = {"phase3": [], "merlin": [], "finals": []}

    def add(key: tuple, record: dict) -> None:
        if key in cells:
            raise ValueError(f"Cell {key} appears twice")
        cells[key] = record

    for group, dirs in (("phase3", phase3_dirs), ("merlin", merlin_dirs)):
        for sweep_dir in dirs:
            status = json.loads((sweep_dir / "sweep_status.json").read_text())
            summary = json.loads((sweep_dir / "summary.json").read_text())
            if summary["coverage"] != "COMPLETE SWEEP":
                raise ValueError(f"{sweep_dir} is not a COMPLETE SWEEP")
            sources[group].append(summary["sweep_id"])
            variant = status["plan"].get("config_variant")
            walls = _legacy_wall_seconds(sweep_dir, status)
            for cell in summary["cells"]:
                arm = _arm(cell["dataset"], variant)
                record = _cell_record(cell, summary["sweep_id"], "A")
                record["wall_seconds_total"] = walls[
                    (arm, cell["model"], cell["bond_dim"])
                ]
                add(
                    (arm, "A", MODEL_LABEL[cell["model"]], cell["bond_dim"], None),
                    record,
                )
            for entry in summary["fig8"]:
                fig8[(_arm(entry["dataset"], variant), entry["bond_dim"])] = entry

    status = json.loads((finals_dir / "sweep_status.json").read_text())
    summary = json.loads((finals_dir / "summary.json").read_text())
    sources["finals"].append(summary["sweep_id"])
    for cell in summary["cells"]:
        protocol = _protocol(cell["config_variant"])
        label = MODEL_LABEL[cell["model"]]
        order = cell["hparams"].get("order") if label == "B5" else None
        add(
            (cell["arm"], protocol, label, cell["bond_dim"], order),
            _cell_record(cell, summary["sweep_id"], protocol),
        )
    return cells, fig8, sources


def _get(cells: dict, arm: str, protocol: str, label: str, h: int, order=None):
    record = cells.get((arm, protocol, label, h, order))
    if record is None or not record["successful_runs"]:
        return None
    return record


def separation(x: dict | None, y: dict | None, split: str) -> dict | None:
    """Return the 2-SEM separation of X against Y; positive difference favours X."""
    if x is None or y is None:
        return None
    difference = y[f"mean_{split}"] - x[f"mean_{split}"]
    margin = SEM_MULTIPLIER * math.hypot(x[f"sem_{split}"], y[f"sem_{split}"])
    return {
        "difference": difference,
        "margin": margin,
        "x_beats_y": difference > margin,
        "y_beats_x": -difference > margin,
    }


def b3_bond_dim(arm: str, h: int) -> int:
    from quantum_correlations_generative_models.lib.basis_enhanced import (
        free_parameter_count,
    )

    alphabet = ALPHABET[arm.replace("_pooled", "")]
    return parameter_matched_hmm_bond_dim(free_parameter_count(h, alphabet), alphabet)


def b3_cell(cells: dict, arm: str, protocol: str, bond_dim: int) -> dict | None:
    """B3 reuses the B1 cell at h' <= 12 and the finals B3 cell above."""
    return _get(cells, arm, protocol, "B1", bond_dim)


def build_rows(cells: dict) -> list[dict]:
    """Return one row per (arm, protocol, split, h) with every model and every flag."""
    rows = []
    for arm in ARMS:
        for protocol in ("A", "B"):
            for split in ("train", "test"):
                for h in BOND_DIMS:
                    h3 = b3_bond_dim(arm, h)
                    models = {
                        "BE": _get(cells, arm, protocol, "BE", h),
                        "B1": _get(cells, arm, protocol, "B1", h),
                        "B2": _get(cells, arm, protocol, "B2", h),
                        "B3": b3_cell(cells, arm, protocol, h3),
                        "B4": _get(cells, arm, protocol, "B4", h),
                    }
                    row = {"arm": arm, "protocol": protocol, "split": split, "bond_dim": h,
                           "b3_bond_dim": h3}  # fmt: skip
                    if protocol == "A":
                        merlin = _get(cells, arm, "A", "MerLin", h)
                        models["MerLin"] = merlin
                        if merlin is not None:
                            alphabet = ALPHABET[arm.replace("_pooled", "")]
                            hm = parameter_matched_hmm_bond_dim(
                                merlin["free_parameter_count"], alphabet
                            )
                            row["b3_merlin_bond_dim"] = hm
                            models["B3_merlin"] = b3_cell(cells, arm, "A", hm)
                    for order in KGRAM_ORDERS:
                        models[f"B5_k{order}"] = _get(
                            cells, arm, protocol, "B5", 0, order
                        )
                    for name, record in models.items():
                        row[f"{name}_runs"] = record["successful_runs"] if record else 0
                        row[f"{name}_mean"] = (
                            record[f"mean_{split}"] if record else None
                        )
                        row[f"{name}_sem"] = record[f"sem_{split}"] if record else None
                        row[f"{name}_best_of_10"] = (
                            record[f"best_of_10_{split}"] if record else None
                        )
                        row[f"{name}_params"] = (
                            record["free_parameter_count"] if record else None
                        )
                    pairs = [("BE", "B1"), ("BE", "B2"), ("BE", "B3"), ("BE", "B4")]
                    pairs += [("BE", f"B5_k{k}") for k in KGRAM_ORDERS]
                    if protocol == "A":
                        pairs += [("MerLin", "B1"), ("MerLin", "B3_merlin")]
                    for x, y in pairs:
                        result = separation(models.get(x), models.get(y), split)
                        row[f"{x}_minus_{y}_advantage"] = (
                            result["difference"] if result else None
                        )
                        row[f"{x}_vs_{y}_margin"] = result["margin"] if result else None
                        row[f"{x}_beats_{y}"] = result["x_beats_y"] if result else None
                        row[f"{y}_beats_{x}"] = result["y_beats_x"] if result else None
                    required = [row[f"BE_beats_{b}"] for b in ("B1", "B2", "B3")]
                    row["BE_beats_B1_B2_B3"] = (
                        None if any(v is None for v in required) else all(required)
                    )
                    separated = []
                    for b in ("B1", "B2", "B3"):
                        advantage = row[f"BE_minus_{b}_advantage"]
                        if advantage is None:
                            separated.append(None)
                            continue
                        separated.append(
                            (row[f"BE_beats_{b}"] or row[f"{b}_beats_BE"])
                            and abs(advantage) > COMPARABLE_TOLERANCE_BITS
                        )
                    row["baseline_separated_by_more_than_0.05"] = (
                        None if any(v is None for v in separated) else any(separated)
                    )
                    rows.append(row)
    return rows


def _rows(rows: list[dict], arm: str, protocol: str, split: str) -> list[dict]:
    return [
        r for r in rows
        if (r["arm"], r["protocol"], r["split"]) == (arm, protocol, split)
    ]  # fmt: skip


def c3_verdict(rows: list[dict], arm: str, protocol: str, split: str) -> dict:
    """Apply the C3 counting rule (and the biofam comparable rule) to one arm."""
    selected = [r for r in _rows(rows, arm, protocol, split) if r["bond_dim"] >= 2]
    missing = [r["bond_dim"] for r in selected if r["BE_beats_B1_B2_B3"] is None]
    holds = [r["bond_dim"] for r in selected if r["BE_beats_B1_B2_B3"]]
    if missing:
        verdict = "UNRESOLVED"
    elif len(holds) >= 9:
        verdict = "SUPPORTED"
    elif len(holds) >= 3:
        verdict = "PARTIALLY_SUPPORTED"
    else:
        verdict = "UNSUPPORTED"
    result = {"arm": arm, "protocol": protocol, "split": split,
              "advantage_holds_at_h": holds, "missing_h": missing,
              "count": len(holds), "verdict": verdict,
              "claim_rule": "counting_rule"}  # fmt: skip
    for b in ("B1", "B2", "B3"):
        result[f"BE_beats_{b}_at_h"] = [
            r["bond_dim"] for r in selected if r[f"BE_beats_{b}"]
        ]
        result[f"{b}_beats_BE_at_h"] = [
            r["bond_dim"] for r in selected if r[f"{b}_beats_BE"]
        ]
    if arm == "biofam":
        late = [r for r in selected if r["bond_dim"] >= 4]
        late_missing = [
            r["bond_dim"]
            for r in late
            if r["baseline_separated_by_more_than_0.05"] is None
        ]
        separated = [
            r["bond_dim"] for r in late if r["baseline_separated_by_more_than_0.05"]
        ]
        result["claim_rule"] = "comparable_rule (the counting verdict is descriptive)"
        result["comparable_rule"] = {
            "separated_by_more_than_0.05_at_h": separated,
            "per_baseline": {
                b: [
                    r["bond_dim"]
                    for r in late
                    if r[f"BE_minus_{b}_advantage"] is not None
                    and (r[f"BE_beats_{b}"] or r[f"{b}_beats_BE"])
                    and abs(r[f"BE_minus_{b}_advantage"]) > COMPARABLE_TOLERANCE_BITS
                ]
                for b in ("B1", "B2", "B3")
            },  # fmt: skip
            "missing_h": late_missing,
            "verdict": "UNRESOLVED"
            if late_missing
            else ("SUPPORTED" if not separated else "UNSUPPORTED"),
        }
    return result


def c4_verdict(cells: dict, fig8: dict, arm: str) -> dict:
    """Keep the Phase 3 5-sigma gap only where BE best-of-10 train < B3 best-of-10 train."""
    entries = []
    for h in BOND_DIMS:
        entry = fig8.get((arm, h))
        enhanced = _get(cells, arm, "A", "BE", h)
        h3 = b3_bond_dim(arm, h)
        matched = b3_cell(cells, arm, "A", h3)
        if entry is None or enhanced is None or matched is None:
            entries.append({"bond_dim": h, "evidence": "EVIDENCE_MISSING"})
            continue
        threshold = entry["lrt_5sigma_threshold_paper_df_bits_per_sequence"]
        gap = entry[f"gap_bits_per_sequence_{METRIC}"]
        five_sigma = threshold is not None and gap > threshold
        below_b3 = enhanced["best_of_10_train"] < matched["best_of_10_train"]
        entries.append({
            "bond_dim": h, "b3_bond_dim": h3,
            "gap_bits_per_sequence": gap, "threshold_5sigma_paper_df": threshold,
            "five_sigma": five_sigma,
            "BE_best_of_10_train": enhanced["best_of_10_train"],
            "B3_best_of_10_train": matched["best_of_10_train"],
            "BE_below_B3": below_b3,
            "advantage_counts": five_sigma and below_b3,
        })  # fmt: skip
    return {
        "arm": arm,
        "five_sigma_at_h": [e["bond_dim"] for e in entries if e.get("five_sigma")],
        "survives_parameter_matching_at_h": [
            e["bond_dim"] for e in entries if e.get("advantage_counts")
        ],
        "attributed_to_parameter_count_at_h": [
            e["bond_dim"]
            for e in entries
            if e.get("five_sigma") and not e["BE_below_B3"]
        ],
        "entries": entries,
    }


def sec6_verdict(rows: list[dict], arm: str) -> dict:
    """BE beats B4 on arm A train and arm B test at >= 3 values of h in 2..12."""
    train = {r["bond_dim"]: r for r in _rows(rows, arm, "A", "train")}
    test = {r["bond_dim"]: r for r in _rows(rows, arm, "B", "test")}
    holds, missing = [], []
    for h in DECISION_BOND_DIMS:
        a, b = train[h]["BE_beats_B4"], test[h]["BE_beats_B4"]
        if a is None or b is None:
            missing.append(h)
        elif a and b:
            holds.append(h)
    if len(holds) >= 3:
        verdict = "SUPPORTED"
    elif len(holds) + len(missing) >= 3:
        verdict = "UNRESOLVED"
    else:
        verdict = "UNSUPPORTED"
    return {
        "arm": arm, "holds_at_h": holds, "missing_h": missing, "verdict": verdict,
        "BE_beats_B4_armA_train_at_h": [h for h in DECISION_BOND_DIMS if train[h]["BE_beats_B4"]],
        "B4_beats_BE_armA_train_at_h": [h for h in DECISION_BOND_DIMS if train[h]["B4_beats_BE"]],
        "BE_beats_B4_armB_test_at_h": [h for h in DECISION_BOND_DIMS if test[h]["BE_beats_B4"]],
        "B4_beats_BE_armB_test_at_h": [h for h in DECISION_BOND_DIMS if test[h]["B4_beats_BE"]],
    }  # fmt: skip


def merlin_summary(rows: list[dict], arm: str) -> dict:
    result = {"arm": arm, "protocol": "A", "label": ROLE["MerLin"]}
    for split in ("train", "test"):
        selected = _rows(rows, arm, "A", split)
        for x, y in (
            ("MerLin", "B1"),
            ("B1", "MerLin"),
            ("MerLin", "B3_merlin"),
            ("B3_merlin", "MerLin"),
        ):
            result[f"{x}_beats_{y}_{split}_at_h"] = [
                r["bond_dim"] for r in selected if r.get(f"{x}_beats_{y}")
            ]
        result[f"missing_{split}_h"] = [
            r["bond_dim"]
            for r in selected
            if r.get("MerLin_beats_B1") is None
            or r.get("MerLin_beats_B3_merlin") is None
        ]
    result["b3_merlin_bond_dims"] = {
        r["bond_dim"]: r.get("b3_merlin_bond_dim")
        for r in _rows(rows, arm, "A", "train")
    }
    return result  # fmt: skip


def kgram_summary(rows: list[dict], arm: str) -> list[dict]:
    out = []
    for protocol in ("A", "B"):
        for split in ("train", "test"):
            selected = _rows(rows, arm, protocol, split)
            for k in KGRAM_ORDERS:
                out.append({
                    "arm": arm, "protocol": protocol, "split": split, "order": k,
                    "B5_mean": selected[0][f"B5_k{k}_mean"],
                    "BE_beats_B5_at_h": [r["bond_dim"] for r in selected if r[f"BE_beats_B5_k{k}"]],
                    "B5_beats_BE_at_h": [r["bond_dim"] for r in selected if r[f"B5_k{k}_beats_BE"]],
                })  # fmt: skip
    return out


PAIR_STATUS = {
    ("A", "BE", "B1"): ("COMMENSURABLE", "Same data, splits, metric and final-epoch checkpoint; equal h (memory axis)."),
    ("A", "BE", "B2"): ("PARTIALLY_COMMENSURABLE", "Same protocol and equal h, but B2's lr is validation-tuned while BE keeps the paper's alpha (selection procedure not stated in the paper); arm B is the matched-tuning comparison."),
    ("A", "BE", "B3"): ("COMMENSURABLE", "Same protocol; B3 has at least BE's free-parameter count (capacity axis)."),
    ("A", "BE", "B4"): ("COMMENSURABLE", "Pure ablation: same optimiser, alpha, epochs and seeds; only the field differs (B4 has fewer parameters)."),
    ("A", "BE", "B5"): ("PARTIALLY_COMMENSURABLE", "Same data, splits and metric, but no capacity match (B5 has no bond dimension); reference only."),
    ("A", "MerLin", "B1"): ("PARTIALLY_COMMENSURABLE", "Same data, splits, metric and checkpoint; MerLin is a simulated adaptation with an untuned lr and heralded post-selection."),
    ("A", "MerLin", "B3_merlin"): ("PARTIALLY_COMMENSURABLE", "As MerLin vs B1, with B3 matched to MerLin's parameter count."),
    ("B", "BE", "B1"): ("COMMENSURABLE", "Same validation carve-out, best-validation checkpoint and tuning budget per model; equal h."),
    ("B", "BE", "B2"): ("COMMENSURABLE", "Same carve-out, checkpoint rule and lr grid; equal h."),
    ("B", "BE", "B3"): ("COMMENSURABLE", "Same carve-out and checkpoint rule; parameter-matched h'."),
    ("B", "BE", "B4"): ("COMMENSURABLE", "Same carve-out, checkpoint rule and lr grid; equal h."),
    ("B", "BE", "B5"): ("PARTIALLY_COMMENSURABLE", "Smoothing tuned on validation, but no capacity match; reference only."),
}  # fmt: skip


def commensurability(rows: list[dict]) -> list[dict]:
    out = []
    for arm in ARMS:
        for (protocol, x, y), (status, reason) in PAIR_STATUS.items():
            column = f"{x}_beats_{y}" if y != "B5" else f"{x}_beats_B5_k1"
            selected = _rows(rows, arm, protocol, "train")
            missing = [r["bond_dim"] for r in selected if r.get(column) is None]
            entry = {"arm": arm, "protocol": protocol, "pair": f"{x} vs {y}",
                     "status": status, "reason": reason, "missing_h": missing}  # fmt: skip
            if missing:
                entry["status"] = (
                    "EVIDENCE_MISSING"
                    if len(missing) == len(selected)
                    else "PARTIALLY_COMMENSURABLE"
                )
                entry["reason"] += (
                    f" Evidence missing at h = {missing} (failed or invalid runs)."
                )
            out.append(entry)
        out.append({"arm": arm, "protocol": "B", "pair": "MerLin vs any", "status": "N/A",
                    "reason": "MerLin is not run in arm B by design (adaptation not tuned; Phase 4 status PARTIAL_MERLIN_TRANSLATION).",
                    "missing_h": []})  # fmt: skip
    return out


def systems(cells: dict) -> list[dict]:
    """Return one row per (arm, protocol, model[, order]) with run counts and wall-clock."""
    grouped: dict[tuple, list] = {}
    for (arm, protocol, label, h, order), record in cells.items():
        name = label if order is None else f"{label}_k{order}"
        if label == "B1" and h > 12:
            name = "B3 (h' > 12)"
        grouped.setdefault((arm, protocol, name), []).append((h, record))
    out = []
    for (arm, protocol, name), items in sorted(grouped.items()):
        records = [r for _, r in items]
        base = name.split("_")[0].split(" ")[0]
        out.append({
            "arm": arm, "protocol": protocol, "system": name,
            "role": ROLE.get("B3" if name.startswith("B3") else base, ""),
            "provenance": "locally executed; sweeps " + ", ".join(sorted({r["sweep_id"] for r in records})),
            "scope": ROLE["MerLin"] if base == "MerLin" else PROTOCOL_SCOPE[protocol],
            "bond_dims": sorted(h for h, _ in items),
            "successful_runs": sum(r["successful_runs"] for r in records),
            "expected_runs": sum(r["expected_runs"] for r in records),
            "invalid_runs": sum(len(r["invalid_runs"]) for r in records),
            "free_parameters_range": [
                min(r["free_parameter_count"] for r in records if r["free_parameter_count"] is not None),
                max(r["free_parameter_count"] for r in records if r["free_parameter_count"] is not None),
            ] if any(r["free_parameter_count"] is not None for r in records) else None,
            "hparams": records[0]["hparams"],
            "wall_seconds_total": sum(r["wall_seconds_total"] or 0.0 for r in records),
        })  # fmt: skip
    return out


def compare(phase3: list[Path], merlin: list[Path], finals: Path) -> dict:
    cells, fig8, sources = load_cells(phase3, merlin, finals)
    rows = build_rows(cells)
    verdicts = {
        "C3_learns_better_armA_train": [c3_verdict(rows, a, "A", "train") for a in ARMS],
        "C3_generalizes_better_armB_test": [c3_verdict(rows, a, "B", "test") for a in ARMS],
        "C5_mean_armA_train": [c3_verdict(rows, a, "A", "train") for a in ARMS],
        "C5_mean_armA_test": [c3_verdict(rows, a, "A", "test") for a in ARMS],
        "descriptive_armB_train": [c3_verdict(rows, a, "B", "train") for a in ARMS],
        "C4": [c4_verdict(cells, fig8, a) for a in ARMS],
        "SecVI_complex_amplitudes": [sec6_verdict(rows, a) for a in ARMS],
        "MerLin_armA": [merlin_summary(rows, a) for a in ARMS],
        "B5_reference": [k for a in ARMS for k in kgram_summary(rows, a)],
    }  # fmt: skip
    return {
        "metric": f"{METRIC} (lower is better); differences equal those of NLL - log2(N)/n",
        "rules": __doc__.split("Writes")[0].strip(),
        "claim_arms": {
            "C3/C5 counting rule": list(C3_ARMS),
            "comparable rule": ["biofam"],
        },
        "sources": sources,
        "verdicts": verdicts,
        "commensurability": commensurability(rows),
        "systems": systems(cells),
        "rows": rows,
        "generated_by": "utils/compare_phase5.py",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase3", type=Path, nargs="+", required=True)
    parser.add_argument("--merlin", type=Path, nargs="+", required=True)
    parser.add_argument("--finals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(
        [p.resolve() for p in args.phase3],
        [p.resolve() for p in args.merlin],
        args.finals.resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    columns = sorted({key for row in result["rows"] for key in row})
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["rows"])
    v = result["verdicts"]
    for name in (
        "C3_learns_better_armA_train",
        "C3_generalizes_better_armB_test",
        "C5_mean_armA_train",
        "C5_mean_armA_test",
        "descriptive_armB_train",
    ):
        for entry in v[name]:
            extra = (
                f" comparable={entry['comparable_rule']}"
                if "comparable_rule" in entry
                else ""
            )
            print(
                f"{name:34s} {entry['arm']:24s} {entry['verdict']:20s} h={entry['advantage_holds_at_h']} missing={entry['missing_h']}{extra}"
            )
    for entry in v["C4"]:
        print(
            f"C4 {entry['arm']:24s} 5sigma={entry['five_sigma_at_h']} survives={entry['survives_parameter_matching_at_h']} param-count={entry['attributed_to_parameter_count_at_h']}"
        )
    for entry in v["SecVI_complex_amplitudes"]:
        print(
            f"SecVI {entry['arm']:24s} {entry['verdict']:12s} h={entry['holds_at_h']} missing={entry['missing_h']}"
        )
    return 0  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
