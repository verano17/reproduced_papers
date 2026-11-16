"""Runtime entry point for the Gao et al. (PRX 12, 021037, 2022) reproduction.

``cfg["experiment"]`` selects one experiment per run:

- ``sequence_model`` (E5): train one model (``model.kind`` =
  ``basis_enhanced`` or ``hmm_baum_welch``; ``photonic_merlin`` is the
  Phase 4 MerLin adaptation, not the paper's model; ``hmm_gradient`` and
  ``kgram_counting`` are the Phase 5 controls B2 and B5) at one bond
  dimension and seed on one dataset (Fig. 7, p14). ``evaluation.checkpoint``
  selects the reported checkpoint (``final`` or ``best_validation``).
- ``theorem1_toy`` (E1 + E3): exact checks of the Fig. 2a model (p4, p7-8)
  and the App. F 6-state hidden process (p23).
- ``robust_l1`` (E2): App. E lemmas (p21-22).
- ``mermin_peres`` (E4): Table I, Eqs. 11-14 and Lemma 1 (p9-10).

Every run writes ``run.log``, ``run_status.json``, ``config_snapshot.json``
(written by the shared runtime, or here when called directly) and
``metrics.json``; ``sequence_model`` runs also write ``model_state.pt``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from runtime_lib.dtypes import dtype_torch

from . import cluster_bbqc, stabilizer
from .basis_enhanced import (
    BasisEnhancedTwoGram,
    free_parameter_count,
    hmm_parameter_count,
)
from .data import dataset_identity, load_splits
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .hmm import HiddenMarkovModel
from .hmm_gradient import GradientHMM
from .kgram import KGramModel
from .photonic_hmm_merlin import COMPUTATION_SPACE, NUM_PHOTONS, PhotonicQuantumHMM
from .training import (
    fit_kgram,
    select_checkpoint,
    train_basis_enhanced,
    train_hmm,
    train_hmm_gradient,
    train_photonic_merlin,
)

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
STATUS_SCHEMA_VERSION = 1
EXACT_SELECTION = {
    "name": "not applicable (exact evaluation, no model selection)",
    "direction": "minimize",
    "split": "none",
    "checkpoint": "none",
}


def _synthetic_identity(description: str) -> dict:
    return {
        "name": description,
        "splits": {},
        "preprocessing": "none",
        "subset": "exact enumeration",
    }


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _candidate(cfg: dict) -> dict:
    """Return the swept values that identify a run (``run_status.json`` and sweeps)."""
    experiment = cfg["experiment"]
    if experiment != "sequence_model":
        return {"experiment": experiment}
    return {
        "experiment": experiment,
        "dataset": cfg["dataset"]["name"],
        "model": cfg["model"]["kind"],
        "field": cfg["model"]["field"],
        "bond_dim": int(cfg["model"]["bond_dim"]),
    }


def run_sequence_model(cfg: dict, run_dir: Path, status: dict) -> dict:
    """Train and evaluate one E5 model; return the metrics mapping."""
    splits = load_splits(cfg)
    checkpoint_rule = cfg["evaluation"]["checkpoint"]
    status["dataset"] = dataset_identity(splits)
    status["selection"] = {
        "name": "nll_bits_per_site",
        "direction": "minimize",
        "split": "train (best-of-trials selection is applied across runs in Phase 3)"
        if checkpoint_rule == "final"
        else "validation (checkpoint and Phase 5 hyperparameter selection)",
        "checkpoint": "last epoch / last EM iteration"
        if checkpoint_rule == "final"
        else "epoch / EM iteration with the lowest validation NLL, ties to the earliest",
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | dataset=%s | train=%d | validation=%d | test=%d | n=%d | M=%d",
        splits.name,
        len(splits.train),
        len(splits.validation),
        len(splits.test),
        splits.sequence_length,
        splits.alphabet_size,
    )
    model_cfg = cfg["model"]
    bond_dim = int(model_cfg["bond_dim"])
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    real_dtype = dtype_torch(cfg["dtype"]) or torch.float64
    device = cfg["device"]
    started = time.perf_counter()
    if model_cfg["kind"] == "basis_enhanced":
        field = model_cfg["field"]
        dtype = _complex_dtype(real_dtype) if field == "complex" else real_dtype
        model = BasisEnhancedTwoGram(
            bond_dim, splits.alphabet_size, generator, dtype
        ).to(device)
        history = train_basis_enhanced(model, splits, cfg)
        parameter_count = free_parameter_count(bond_dim, splits.alphabet_size, field)
        state = model.state_dict()
    elif model_cfg["kind"] == "hmm_baum_welch":
        model = HiddenMarkovModel.random(bond_dim, splits.alphabet_size, generator).to(
            device
        )
        model, history = train_hmm(model, splits, cfg)
        parameter_count = hmm_parameter_count(bond_dim, splits.alphabet_size)
        state = {
            "initial": model.initial,
            "transition": model.transition,
            "emission": model.emission,
        }
    elif model_cfg["kind"] == "photonic_merlin":
        if device != "cpu":
            raise ValueError("photonic_merlin is simulated on CPU in this adaptation")
        model = PhotonicQuantumHMM(bond_dim, splits.alphabet_size, int(cfg["seed"]))
        history = train_photonic_merlin(model, splits, cfg)
        parameter_count = model.trainable_parameter_count()
        state = model.state_dict()
        photonic_record = {
            "label": "MerLin adaptation (simulated), not the paper's model",
            "computation_space": str(COMPUTATION_SPACE),
            "detector_model": "photon-number-resolving on the data modes; hidden modes not measured",
            "photon_number": NUM_PHOTONS,
            "num_modes": model.num_modes,
            "input_state": f"memory photon in hidden modes 0..{bond_dim - 1} (trained amplitudes) + probe photon in mode {model.probe_mode} at every site",
            "encoding": "symbol = data mode of the heralded photon; no classical input encoding",
            "measurement_strategy": "MeasurementStrategy.amplitudes(FOCK), heralded data-mode outcomes",
            "postselection": "exactly one photon in the data modes at every site (global post-selection)",
            "simulator": "MerLin 0.4.1 exact CPU simulation, shots = 0",
            "herald_success_log2_per_sequence": history[-1][
                "herald_success_log2_per_sequence"
            ],
        }
    elif model_cfg["kind"] == "hmm_gradient":
        model = GradientHMM(bond_dim, splits.alphabet_size, generator).to(device)
        history = train_hmm_gradient(model, splits, cfg)
        parameter_count = hmm_parameter_count(bond_dim, splits.alphabet_size)
        state = model.state_dict()
    elif model_cfg["kind"] == "kgram_counting":
        model = KGramModel(
            int(model_cfg["order"]),
            splits.alphabet_size,
            float(cfg["training"]["smoothing"]),
        )
        history = fit_kgram(model, splits, cfg)
        parameter_count = model.free_parameter_count()
        state = {"tables": [torch.as_tensor(table) for table in model.tables]}
    else:
        raise ValueError(f"Unknown model.kind {model_cfg['kind']!r}")
    checkpoint = select_checkpoint(history, checkpoint_rule)
    logger.info(
        "CHECKPOINT_SELECTED | rule=%s | step=%s",
        checkpoint["rule"],
        checkpoint["step"],
    )
    torch.save(state, run_dir / "model_state.pt")
    logger.info("ARTIFACT_WRITTEN | path=model_state.pt")
    extra = (
        {"photonic": photonic_record} if model_cfg["kind"] == "photonic_merlin" else {}
    )
    if model_cfg["kind"] == "kgram_counting":
        # bond_dim is not a parameter of a k-gram model; its configs set it to 0.
        extra["kgram"] = {"order": model.order, "smoothing": model.smoothing}
    return {
        **extra,
        "experiment": "sequence_model",
        "dataset": splits.name,
        "model_kind": model_cfg["kind"],
        "field": model_cfg["field"],
        "bond_dim": bond_dim,
        "seed": int(cfg["seed"]),
        "sequence_length": splits.sequence_length,
        "alphabet_size": splits.alphabet_size,
        "free_parameter_count": parameter_count,
        "wall_seconds": time.perf_counter() - started,
        "final": history[-1]["evaluation"],
        "checkpoint": checkpoint,
        "history": history,
    }


def run_theorem1_toy(cfg: dict, run_dir: Path, status: dict) -> dict:
    """E1 + E3: exact Fig. 2a distribution, Eq. 4/5, local supports, k-gram fits, App. F."""
    settings = cfg["theorem1"]
    status["dataset"] = _synthetic_identity(
        "Fig. 2a cluster-state basis-enhanced 2-gram model (exact)"
    )
    status["selection"] = EXACT_SELECTION
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | dataset=exact model distribution | num_pairs=%s",
        settings["num_pairs"],
    )
    check_pairs = int(settings["statevector_check_pairs"])
    logger.info(
        "EVALUATION_STARTED | check=statevector_cross_check | num_pairs=%d", check_pairs
    )
    transfer = cluster_bbqc.exact_distribution(check_pairs)
    circuit = cluster_bbqc.statevector_distribution(check_pairs)
    conditional = cluster_bbqc.ghz_conditional(transfer, list(settings["chosen_pairs"]))
    toy = {
        "statevector_max_abs_difference": float(np.abs(transfer - circuit).max()),
        "eq4_real_part_max_abs_difference": float(
            np.abs(conditional - cluster_bbqc.eq4_reference(True)).max()
        ),
        "eq4_modulus_max_abs_difference": float(
            np.abs(conditional - cluster_bbqc.eq4_reference(False)).max()
        ),
        "eq5_support_violations": cluster_bbqc.eq5_violations(conditional),
        "local_product_supports": cluster_bbqc.local_support_matches(conditional),
    }
    logger.info(
        "EVALUATION_COMPLETED | check=statevector_cross_check | statevector_diff=%.2e | eq4_real_part_diff=%.2e | eq5_violations=%d | local_support_matches=%d",
        toy["statevector_max_abs_difference"],
        toy["eq4_real_part_max_abs_difference"],
        toy["eq5_support_violations"],
        toy["local_product_supports"]["exact_matches"],
    )
    per_size = []
    for num_pairs in settings["num_pairs"]:
        logger.info(
            "EVALUATION_STARTED | check=kgram_and_hidden_process | num_pairs=%d",
            num_pairs,
        )
        target = cluster_bbqc.exact_distribution(num_pairs)
        hidden = cluster_bbqc.hidden_process_distribution(num_pairs)
        fits = []
        for order in range(1, num_pairs + 1):
            comparison = cluster_bbqc.compare_distributions(
                cluster_bbqc.ml_kgram(target, order), target
            )
            fits.append({"order": order, **comparison})
            logger.info(
                "KGRAM_FIT | num_pairs=%d | k=%d | support_equal=%s | outside_mass=%.6f | l1=%.6f",
                num_pairs,
                order,
                comparison["support_equal"],
                comparison["model_mass_outside_target_support"],
                comparison["l1_distance"],
            )
        minimal_order = next(
            (fit["order"] for fit in fits if fit["support_equal"]), None
        )
        entry = {
            "num_pairs": num_pairs,
            "theorem1_threshold_floor_n_minus_1_over_2": (num_pairs - 1) // 2,
            "minimal_ml_kgram_order_with_equal_support": minimal_order,
            "six_state_hidden_process_max_abs_difference": float(
                np.abs(hidden - target).max()
            ),
            "kgram_fits": fits,
        }
        per_size.append(entry)
        logger.info(
            "EVALUATION_COMPLETED | check=kgram_and_hidden_process | num_pairs=%d | minimal_k=%s | hidden_process_diff=%.2e",
            num_pairs,
            minimal_order,
            entry["six_state_hidden_process_max_abs_difference"],
        )
    return {"experiment": "theorem1_toy", "toy_model": toy, "per_size": per_size}


def run_robust_l1(cfg: dict, run_dir: Path, status: dict) -> dict:
    """E2: App. E random walk, Lemma 4 and 6 numbers, and l1 of ML k-gram fits."""
    settings = cfg["robust_l1"]
    status["dataset"] = _synthetic_identity("App. E S3 walk and Fig. 2a model (exact)")
    status["selection"] = EXACT_SELECTION
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | dataset=exact enumeration | num_pairs=%s",
        settings["num_pairs"],
    )
    logger.info("EVALUATION_STARTED | check=s3_walk")
    matrix = stabilizer.s3_walk_transfer_matrix()
    walk = [
        {
            "steps": steps,
            "probability_identity_or_s": stabilizer.identity_or_s_probability(steps),
            "eq_e5": 1 / 3 + (2 / 3) * 4.0**-steps,
            "lemma4_ghz_type_probability": stabilizer.identity_or_s_probability(steps)
            ** 2,
        }
        for steps in settings["walk_steps"]
    ]
    walk_result = {
        "transfer_matrix_times_4": (matrix * 4).tolist(),
        "matches_printed_e4_up_to_relabelling": stabilizer.matches_printed_e4(matrix),
        "walk": walk,
        "lemma6_lhv_max_success": stabilizer.ghz_lhv_max_success(),
    }
    logger.info(
        "EVALUATION_COMPLETED | check=s3_walk | matches_e4=%s | lhv_max=%.6f",
        walk_result["matches_printed_e4_up_to_relabelling"],
        walk_result["lemma6_lhv_max_success"],
    )
    l1_fits = []
    for num_pairs in settings["num_pairs"]:
        logger.info("EVALUATION_STARTED | check=kgram_l1 | num_pairs=%d", num_pairs)
        target = cluster_bbqc.exact_distribution(num_pairs)
        for order in settings["kgram_orders"]:
            l1 = cluster_bbqc.compare_distributions(
                cluster_bbqc.ml_kgram(target, order), target
            )["l1_distance"]
            l1_fits.append(
                {
                    "num_pairs": num_pairs,
                    "order": order,
                    "l1_distance": l1,
                    "bound_1_over_288": 1 / 288,
                }
            )
            logger.info(
                "EVALUATION_COMPLETED | check=kgram_l1 | num_pairs=%d | k=%d | l1=%.6f",
                num_pairs,
                order,
                l1,
            )
    return {"experiment": "robust_l1", "s3_walk": walk_result, "ml_kgram_l1": l1_fits}


def run_mermin_peres(cfg: dict, run_dir: Path, status: dict) -> dict:
    """E4: Table I relations, Eqs. 11-14 and stabilizer counts."""
    settings = cfg["mermin_peres"]
    status["dataset"] = _synthetic_identity(
        "Mermin-Peres square and stabilizer states (exact)"
    )
    status["selection"] = EXACT_SELECTION
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | dataset=exact enumeration | count_qubits=%s",
        settings["count_qubits"],
    )
    logger.info("EVALUATION_STARTED | check=mermin_peres_square")
    square = stabilizer.mermin_peres_checks()
    logger.info(
        "EVALUATION_COMPLETED | check=mermin_peres_square | column_signs=%s",
        square["column_product_signs"],
    )
    counts = []
    for num_qubits in settings["count_qubits"]:
        logger.info("EVALUATION_STARTED | check=stabilizer_count | n=%d", num_qubits)
        enumerated = stabilizer.enumerate_stabilizer_states(num_qubits)
        formula = stabilizer.stabilizer_count_formula(num_qubits)
        counts.append(
            {
                "num_qubits": num_qubits,
                "enumerated": enumerated,
                "formula": formula,
                "log2_count": math.log2(enumerated),
                "n_squared_over_2": num_qubits**2 / 2,
            }
        )
        logger.info(
            "EVALUATION_COMPLETED | check=stabilizer_count | n=%d | enumerated=%d | formula=%d",
            num_qubits,
            enumerated,
            formula,
        )
    return {"experiment": "mermin_peres", "square": square, "stabilizer_counts": counts}


EXPERIMENTS = {
    "sequence_model": run_sequence_model,
    "theorem1_toy": run_theorem1_toy,
    "robust_l1": run_robust_l1,
    "mermin_peres": run_mermin_peres,
}


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the configured experiment and write its evidence to ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration after defaults, overlays and CLI flags.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    ValueError
        If ``cfg["experiment"]`` is unknown.
    """
    run_dir = Path(run_dir)
    configure_logging(
        run_dir / "run.log",
        level=getattr(logging, str(cfg["logging"]["level"]).upper()),
    )
    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    experiment = cfg["experiment"]
    status = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": _candidate(cfg),
        "seed": cfg["seed"],
        "repetition": cfg["seed"],
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": {
            "name": "not loaded yet",
            "splits": {},
            "preprocessing": "not loaded yet",
            "subset": "not loaded yet",
        },
        "selection": {**EXACT_SELECTION, "name": "not set yet"},
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path),
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | seed=%s",
        run_dir.name,
        experiment,
        cfg["seed"],
    )
    try:
        if experiment not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment {experiment!r}; expected one of {sorted(EXPERIMENTS)}"
            )
        metrics = EXPERIMENTS[experiment](cfg, run_dir, status)
        write_json(run_dir / "metrics.json", metrics)
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_sha256=sha256(run_dir / "metrics.json"),
        )
        write_json(run_dir / "run_status.json", status)
        logger.info("RUN_COMPLETED | run_id=%s", run_dir.name)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", run_dir.name)
        raise
