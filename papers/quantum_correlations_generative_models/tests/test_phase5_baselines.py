"""Phase 5 controls: B2 gradient HMM, B3 parameter matching, B4 real BE, B5 k-grams.

Also the best-validation checkpoint rule and the sweep spec expansion used by
the Phase 5 sweeps (LOG.md, Phase 5 Comparison Plan).
"""

from __future__ import annotations

import itertools
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from common import PROJECT_DIR, load_runtime_ready_config
from quantum_correlations_generative_models.lib import runner
from quantum_correlations_generative_models.lib.basis_enhanced import (
    free_parameter_count,
    hmm_parameter_count,
    parameter_matched_hmm_bond_dim,
)
from quantum_correlations_generative_models.lib.hmm import HiddenMarkovModel
from quantum_correlations_generative_models.lib.hmm_gradient import GradientHMM
from quantum_correlations_generative_models.lib.kgram import KGramModel
from quantum_correlations_generative_models.lib.training import select_checkpoint

from runtime_lib.config import deep_update

sys.path.insert(0, str(PROJECT_DIR / "utils"))
import run_sweep  # noqa: E402


def _all_sequences(alphabet: int, length: int) -> np.ndarray:
    return np.array(list(itertools.product(range(alphabet), repeat=length)))


def test_gradient_hmm_equals_softmaxed_hmm_and_is_normalised():
    model = GradientHMM(3, 2, torch.Generator().manual_seed(0))
    sequences = torch.as_tensor(_all_sequences(2, 5))
    with torch.no_grad():
        log_probs = model.log_prob(sequences)
        reference = model.as_hmm().log_prob(sequences)
    torch.testing.assert_close(log_probs, reference)
    assert float(log_probs.exp().sum()) == pytest.approx(1.0, abs=1e-12)


def test_gradient_hmm_starts_at_the_baum_welch_initialisation():
    gradient = GradientHMM(4, 3, torch.Generator().manual_seed(7)).as_hmm()
    baum_welch = HiddenMarkovModel.random(4, 3, torch.Generator().manual_seed(7))
    for name in ("initial", "transition", "emission"):
        torch.testing.assert_close(
            getattr(gradient, name).detach(), getattr(baum_welch, name)
        )


def test_gradient_hmm_gradients_reach_every_logit():
    model = GradientHMM(3, 2, torch.Generator().manual_seed(1))
    loss = -model.log_prob(torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]])).mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("order", [1, 2, 3])
def test_kgram_is_normalised(order):
    rng = np.random.default_rng(order)
    model = KGramModel(order, 3, smoothing=0.5).fit(rng.integers(0, 3, (40, 5)))
    total = np.exp(model.log_prob(_all_sequences(3, 5))).sum()
    assert total == pytest.approx(1.0, abs=1e-12)


def test_kgram_order_one_is_the_smoothed_marginal_and_counts_parameters():
    train = np.array([[0, 0, 1], [0, 2, 0]])
    model = KGramModel(1, 3, smoothing=1.0).fit(train)
    # counts 0:4, 1:1, 2:1 plus 1 each -> 5/9, 2/9, 2/9
    np.testing.assert_allclose(model.tables[0][0], [5 / 9, 2 / 9, 2 / 9])
    assert model.free_parameter_count() == 2
    assert KGramModel(3, 4, 0.1).free_parameter_count() == 3 + 4 * 3 + 16 * 3


def test_kgram_order_two_learns_a_deterministic_chain():
    train = np.array([[0, 1, 0, 1, 0, 1]] * 20 + [[1, 0, 1, 0, 1, 0]] * 20)
    model = KGramModel(2, 2, smoothing=0.01).fit(train)
    log_probs = model.log_prob(train[:1])
    assert float(np.exp(log_probs[0])) > 0.45  # ~0.5 x (alternation ~ 1)


def test_kgram_rejects_invalid_settings():
    with pytest.raises(ValueError):
        KGramModel(0, 2, 0.5)
    with pytest.raises(ValueError):
        KGramModel(2, 2, 0.0)


@pytest.mark.parametrize(
    "alphabet,expected",
    [(8, [1, 4, 7, 9, 12, 15, 18, 21, 24, 27, 30, 33]),
     (4, [1, 4, 6, 8, 10, 12, 15, 17, 19, 21, 24, 26]),
     (2, [1, 3, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21])],
)  # fmt: skip
def test_parameter_matched_hmm_sizes_are_the_planned_ones(alphabet, expected):
    sizes = [
        parameter_matched_hmm_bond_dim(free_parameter_count(h, alphabet), alphabet)
        for h in range(1, 13)
    ]
    assert sizes == expected
    for h, size in zip(range(1, 13), sizes, strict=True):
        target = free_parameter_count(h, alphabet)
        assert hmm_parameter_count(size, alphabet) >= target
        assert size == 1 or hmm_parameter_count(size - 1, alphabet) < target


def test_select_checkpoint_rules():
    history = [
        {"epoch": e, "evaluation": {"validation": {"nll_bits_per_site": v}}}
        for e, v in enumerate([3.0, 2.0, 1.5, 1.5, 1.7])
    ]
    assert select_checkpoint(history, "final")["step"] == 4
    assert select_checkpoint(history, "best_validation")["step"] == 2
    with pytest.raises(ValueError):
        select_checkpoint([{"epoch": 0, "evaluation": {}}], "best_validation")
    with pytest.raises(ValueError):
        select_checkpoint(history, "best_test")


def _sequence_cfg(overlay: dict) -> dict:
    cfg = deep_update(load_runtime_ready_config(), overlay)
    cfg.setdefault("logging", {"level": "info"})
    return cfg


@pytest.mark.parametrize(
    "model,training",
    [
        ({"kind": "hmm_gradient", "bond_dim": 3, "field": "real"},
         {"epochs": 2, "lr": 0.05, "momentum": 0.5}),
        ({"kind": "kgram_counting", "bond_dim": 0, "field": "real", "order": 2},
         {"epochs": 0, "smoothing": 0.1}),
        ({"kind": "basis_enhanced", "bond_dim": 2, "field": "real"},
         {"epochs": 2, "lr": 0.01, "momentum": 0.5}),
    ],
)  # fmt: skip
def test_runner_trains_phase5_controls_with_best_validation_checkpoint(
    tmp_path, model, training
):
    cfg = _sequence_cfg(
        {
            "dataset": {"validation_fraction": 0.2},
            "model": model,
            "training": training,
            "evaluation": {"checkpoint": "best_validation"},
        }
    )
    runner.train_and_evaluate(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["dataset"]["splits"]["validation"] == 16  # floor(80 * 0.2)
    assert metrics["model_kind"] == model["kind"]
    assert metrics["field"] == model["field"]
    checkpoint = metrics["checkpoint"]
    assert checkpoint["rule"] == "best_validation"
    values = [
        entry["evaluation"]["validation"]["nll_bits_per_site"]
        for entry in metrics["history"]
    ]
    assert checkpoint["index"] == int(np.argmin(values))
    if model["kind"] == "kgram_counting":
        assert metrics["kgram"] == {"order": 2, "smoothing": 0.1}
        assert metrics["free_parameter_count"] == 1 + 2
    if model["kind"] == "basis_enhanced":
        assert metrics["free_parameter_count"] == free_parameter_count(2, 2, "real")


def test_default_checkpoint_is_final(tmp_path):
    cfg = _sequence_cfg({"training": {"epochs": 1}})
    runner.train_and_evaluate(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["checkpoint"]["rule"] == "final"
    assert metrics["checkpoint"]["evaluation"] == metrics["final"]


def test_sweep_spec_expands_grids_bond_dims_and_kgram():
    blocks = [
        {
            "datasets": ["biofam", "spect_heart"],
            "config_variant": "_val",
            "model_kinds": ["basis_enhanced", "kgram_counting"],
            "bond_dims": {"biofam": [4, 8], "spect_heart": [4]},
            "seeds": [0, 1],
            "grid": {"lr": [0.001, 0.01]},
        }
    ]
    runs = run_sweep.planned_runs(blocks)
    basis = [r for r in runs if r["kind"] == "basis_enhanced"]
    kgram = [r for r in runs if r["kind"] == "kgram_counting"]
    assert len(basis) == (2 + 1) * 2 * 2
    assert len(kgram) == 2 * 2 * 2 and {r["bond_dim"] for r in kgram} == {0}
    assert {r["hparams"]["lr"] for r in runs} == {0.001, 0.01}
    with pytest.raises(ValueError):
        run_sweep.planned_runs(blocks + blocks)


def test_sweep_spec_files_reference_existing_configs():
    for spec_path in sorted((PROJECT_DIR / "configs" / "sweeps").glob("*.json")):
        blocks = run_sweep.blocks_from_args(SimpleNamespace(spec=str(spec_path)))
        for run in run_sweep.planned_runs(blocks):
            config = (
                PROJECT_DIR
                / "configs"
                / f"e5_fig7_{run_sweep.DATASET_CONFIG[run['dataset']]}"
                f"{run['config_variant']}_{run_sweep.KIND_SUFFIX[run['kind']]}.json"
            )
            assert config.exists(), (spec_path.name, config.name)


def test_tuning_tie_break_follows_the_plan():
    import select_hyperparameters as selection

    assert selection._tie_break([0.001, 0.01], 0.01) == 0.01
    assert selection._tie_break([0.003, 0.03], 0.01) == 0.03  # 3x closer than 3.33x
    assert (
        selection._tie_break([0.001, 0.1], 0.01) == 0.001
    )  # equal log distance -> smaller
    assert selection._tie_break([0.03], 0.001) == 0.03
    assert selection._tie_break([0.001, 0.01], 0.0) == 0.001


def test_finals_spec_has_the_planned_run_count():
    import select_hyperparameters as selection

    arms = ["biofam", "promoter_gene_sequences", "spect_heart", "spect_heart_pooled"]
    models = ["basis_enhanced", "basis_enhanced_real", "hmm_gradient", "hmm_baum_welch"]
    selections = [
        {"arm": arm, "model": model, "order": None, "selected": 0.01}
        for arm in arms
        for model in models
    ] + [
        {"arm": arm, "model": "kgram_counting", "order": order, "selected": 0.5}
        for arm in arms
        for order in (1, 2, 3)
    ]
    spec = selection.finals_spec({"tuning_sweep": "test", "selections": selections})
    assert len(run_sweep.planned_runs(spec["blocks"])) == 1282 + 2162


def _verdict_rows(holds_at: set[int], missing_at: set[int] = frozenset()) -> list[dict]:
    rows = []
    for protocol, split in (("A", "train"), ("B", "test")):
        for h in range(1, 13):
            beats = None if h in missing_at else h in holds_at
            rows.append({"arm": "promoter_gene_sequences", "protocol": protocol,
                         "split": split, "bond_dim": h, "BE_beats_B1_B2_B3": beats,
                         "BE_beats_B4": beats, "B4_beats_BE": False if beats is not None else None,
                         **{f"BE_beats_{b}": beats for b in ("B1", "B2", "B3")},
                         **{f"{b}_beats_BE": False for b in ("B1", "B2", "B3")}})  # fmt: skip
    return rows


def test_separation_rule_is_two_combined_sems():
    import compare_phase5 as comparison

    x = {"mean_train": 1.0, "sem_train": 0.03}
    y = {"mean_train": 1.11, "sem_train": 0.04}  # margin 2 * 0.05 = 0.1
    result = comparison.separation(x, y, "train")
    assert result["x_beats_y"] and not result["y_beats_x"]
    assert result["margin"] == pytest.approx(0.1)
    y["mean_train"] = 1.09
    assert not comparison.separation(x, y, "train")["x_beats_y"]
    assert comparison.separation(x, None, "train") is None


@pytest.mark.parametrize(
    "holds,missing,expected",
    [(set(range(2, 11)), set(), "SUPPORTED"), (set(range(2, 10)), set(), "PARTIALLY_SUPPORTED"),
     ({2, 3, 4}, set(), "PARTIALLY_SUPPORTED"), ({2, 3}, set(), "UNSUPPORTED"),
     (set(range(2, 13)), {5}, "UNRESOLVED"), (set(), {1}, "UNSUPPORTED")],
)  # fmt: skip
def test_c3_counting_rule_follows_the_plan(holds, missing, expected):
    import compare_phase5 as comparison

    rows = _verdict_rows(holds, missing)
    verdict = comparison.c3_verdict(rows, "promoter_gene_sequences", "A", "train")
    assert verdict["verdict"] == expected


@pytest.mark.parametrize(
    "holds,missing,expected",
    [({2, 3, 4}, set(), "SUPPORTED"), ({2, 3}, {4}, "UNRESOLVED"), ({2}, {3}, "UNSUPPORTED")],
)  # fmt: skip
def test_sec6_rule_is_unresolved_only_when_missing_cells_could_matter(
    holds, missing, expected
):
    import compare_phase5 as comparison

    rows = _verdict_rows(holds, missing)
    assert (
        comparison.sec6_verdict(rows, "promoter_gene_sequences")["verdict"] == expected
    )


def test_b3_reuses_the_planned_matched_sizes():
    import compare_phase5 as comparison

    assert [comparison.b3_bond_dim("spect_heart_pooled", h) for h in range(1, 13)] == [
        1, 3, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21
    ]  # fmt: skip
