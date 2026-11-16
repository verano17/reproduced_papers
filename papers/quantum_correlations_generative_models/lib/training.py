"""Training and evaluation loops for the Sec. V experiments (p12-13).

Basis-enhanced model: minibatch Riemannian momentum descent (Eqs. 22-26) on
the loss -(1/K) sum_batch ln p(x), the minibatch form of Eq. B7 (p19) whose
gradient is Eq. 20 (p12). Classical HMM: full-batch Baum-Welch iterations.
MerLin adaptation (Phase 4): the same minibatch loss, optimised by Adam on
the MZI phases and initial-memory amplitudes. Phase 5 controls: B2, the HMM
trained by the basis-enhanced model's minibatch protocol (SGD with momentum on
softmax logits); B5, k-gram counting models (no iterations).
Every split is evaluated before training (epoch 0) and after every epoch or
EM iteration, so the per-epoch history in ``metrics.json`` supports any
checkpoint rule decided later; the reported final values are the last epoch.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch

from .basis_enhanced import BasisEnhancedTwoGram, RiemannianMomentum, unitarity_error
from .data import SequenceSplits, build_dataloader
from .hmm import HiddenMarkovModel
from .hmm_gradient import GradientHMM
from .kgram import KGramModel
from .metrics import sequence_metrics
from .photonic_hmm_merlin import PhotonicQuantumHMM, herald_success_bits

logger = logging.getLogger(__name__)
EVALUATED_SPLITS = ("train", "validation", "test")


def evaluate_splits(
    log_prob_fn, splits: SequenceSplits, device: str, step: int
) -> dict:
    """Evaluate every non-empty split and log each evaluation.

    Parameters
    ----------
    log_prob_fn : callable
        Maps a ``(batch, n)`` long tensor to natural-log probabilities.
    splits : SequenceSplits
        Data splits.
    device : str
        Torch device for the evaluation tensors.
    step : int
        Epoch or EM iteration index (0 = before training).
    """
    results = {}
    for split in EVALUATED_SPLITS:
        sequences = splits.split(split)
        if len(sequences) == 0:
            continue
        logger.info("EVALUATION_STARTED | split=%s | step=%d", split, step)
        with torch.no_grad():
            log_probs = log_prob_fn(
                torch.as_tensor(sequences, dtype=torch.long, device=device)
            )
        values = log_probs.detach().cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                f"Non-finite log-probability on {split} at step {step}"
            )
        results[split] = sequence_metrics(values, sequences)
        logger.info(
            "EVALUATION_COMPLETED | split=%s | step=%d | nll_bits_per_site=%.6f | kl_to_empirical_bits_per_site=%.6f",
            split,
            step,
            results[split]["nll_bits_per_site"],
            results[split]["kl_to_empirical_bits_per_site"],
        )
    return results


def train_basis_enhanced(
    model: BasisEnhancedTwoGram, splits: SequenceSplits, cfg: dict
) -> list[dict]:
    """Train the basis-enhanced model and return the per-epoch history."""
    training = cfg["training"]
    epochs = int(training["epochs"])
    optimizer = RiemannianMomentum(
        model.unitary_parameters(), lr=training["lr"], momentum=training["momentum"]
    )
    loader = build_dataloader(cfg, "train", splits)
    device = cfg["device"]
    history = [
        {"epoch": 0, "evaluation": evaluate_splits(model.log_prob, splits, device, 0)}
    ]
    logger.info(
        "TRAINING_STARTED | model=basis_enhanced | epochs=%d | batches_per_epoch=%d | lr=%g | momentum=%g",
        epochs,
        len(loader),
        training["lr"],
        training["momentum"],
    )
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        batch_losses = []
        for batch_index, (sequences,) in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss = -model.log_prob(sequences).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
            logger.debug("batch=%d | loss_nats=%.6f", batch_index, batch_losses[-1])
        evaluation = evaluate_splits(model.log_prob, splits, device, epoch)
        drift = unitarity_error(model)
        mean_loss_bits = float(np.mean(batch_losses)) / (
            splits.sequence_length * math.log(2.0)
        )
        history.append(
            {
                "epoch": epoch,
                "mean_minibatch_nll_bits_per_site": mean_loss_bits,
                "unitarity_error": drift,
                "seconds": time.perf_counter() - started,
                "evaluation": evaluation,
            }
        )
        logger.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | minibatch_nll_bits_per_site=%.6f | train_nll_bits_per_site=%.6f | unitarity_error=%.2e | seconds=%.2f",
            epoch,
            epochs,
            mean_loss_bits,
            evaluation["train"]["nll_bits_per_site"],
            drift,
            history[-1]["seconds"],
        )
    return history


def train_hmm(
    model: HiddenMarkovModel, splits: SequenceSplits, cfg: dict
) -> tuple[HiddenMarkovModel, list[dict]]:
    """Run Baum-Welch and return the final model and per-iteration history."""
    training = cfg["training"]
    iterations = int(training["em_iterations"])
    tolerance = float(training["em_tolerance"])
    pseudo_count = float(training["pseudo_count"])
    device = cfg["device"]
    train_sequences = torch.as_tensor(splits.train, dtype=torch.long, device=device)
    history = [
        {
            "iteration": 0,
            "evaluation": evaluate_splits(model.log_prob, splits, device, 0),
        }
    ]
    logger.info(
        "TRAINING_STARTED | model=hmm_baum_welch | max_iterations=%d | tolerance=%g | pseudo_count=%g",
        iterations,
        tolerance,
        pseudo_count,
    )
    previous = history[0]["evaluation"]["train"]["nll_bits_per_site"]
    stop_reason = "iteration cap reached"
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        model = model.baum_welch_step(train_sequences, pseudo_count)
        evaluation = evaluate_splits(model.log_prob, splits, device, iteration)
        current = evaluation["train"]["nll_bits_per_site"]
        history.append(
            {
                "iteration": iteration,
                "seconds": time.perf_counter() - started,
                "evaluation": evaluation,
            }
        )
        logger.info(
            "TRAIN_ITERATION_COMPLETED | iteration=%d/%d | train_nll_bits_per_site=%.8f | seconds=%.3f",
            iteration,
            iterations,
            current,
            history[-1]["seconds"],
        )
        if abs(previous - current) <= tolerance * abs(previous):
            stop_reason = f"relative train NLL change <= {tolerance:g}"
            break
        previous = current
    logger.info(
        "TRAINING_STOPPED | model=hmm_baum_welch | iterations=%d | reason=%s",
        len(history) - 1,
        stop_reason,
    )
    return model, history


def train_photonic_merlin(
    model: PhotonicQuantumHMM, splits: SequenceSplits, cfg: dict
) -> list[dict]:
    """Train the MerLin photonic quantum HMM and return the per-epoch history.

    Same minibatch loss, epochs and evaluation schedule as
    :func:`train_basis_enhanced`; the optimiser is Adam because the mesh is
    parameterised by unconstrained phases (Phase 4 adaptation plan).
    """
    training = cfg["training"]
    epochs = int(training["epochs"])
    if training["optimizer"] != "adam":
        raise ValueError(
            f"photonic_merlin supports optimizer 'adam', got {training['optimizer']!r}"
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["lr"]))
    loader = build_dataloader(cfg, "train", splits)
    device = cfg["device"]
    length = splits.sequence_length
    history = [
        {
            "epoch": 0,
            "herald_success_log2_per_sequence": herald_success_bits(model, length),
            "evaluation": evaluate_splits(model.log_prob, splits, device, 0),
        }
    ]
    logger.info(
        "TRAINING_STARTED | model=photonic_merlin | epochs=%d | batches_per_epoch=%d | optimizer=adam | lr=%g",
        epochs,
        len(loader),
        training["lr"],
    )
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        batch_losses = []
        for batch_index, (sequences,) in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss = -model.log_prob(sequences).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
            logger.debug("batch=%d | loss_nats=%.6f", batch_index, batch_losses[-1])
        evaluation = evaluate_splits(model.log_prob, splits, device, epoch)
        success_bits = herald_success_bits(model, length)
        mean_loss_bits = float(np.mean(batch_losses)) / (length * math.log(2.0))
        history.append(
            {
                "epoch": epoch,
                "mean_minibatch_nll_bits_per_site": mean_loss_bits,
                "herald_success_log2_per_sequence": success_bits,
                "seconds": time.perf_counter() - started,
                "evaluation": evaluation,
            }
        )
        logger.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | minibatch_nll_bits_per_site=%.6f | train_nll_bits_per_site=%.6f | herald_success_log2=%.3f | seconds=%.2f",
            epoch,
            epochs,
            mean_loss_bits,
            evaluation["train"]["nll_bits_per_site"],
            success_bits,
            history[-1]["seconds"],
        )
    return history


def train_hmm_gradient(
    model: GradientHMM, splits: SequenceSplits, cfg: dict
) -> list[dict]:
    """Train the B2 HMM by minibatch SGD with momentum; return the per-epoch history.

    Same loss, minibatch size, epochs, shuffling and evaluation schedule as
    :func:`train_basis_enhanced`; the momentum update is Euclidean on the
    softmax logits (the HMM rows are not unitary).
    """
    training = cfg["training"]
    epochs = int(training["epochs"])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["lr"]),
        momentum=float(training["momentum"]),
    )
    loader = build_dataloader(cfg, "train", splits)
    device = cfg["device"]
    history = [
        {"epoch": 0, "evaluation": evaluate_splits(model.log_prob, splits, device, 0)}
    ]
    logger.info(
        "TRAINING_STARTED | model=hmm_gradient | epochs=%d | batches_per_epoch=%d | optimizer=sgd | lr=%g | momentum=%g",
        epochs,
        len(loader),
        training["lr"],
        training["momentum"],
    )
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        batch_losses = []
        for batch_index, (sequences,) in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss = -model.log_prob(sequences).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_index}"
                )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
            logger.debug("batch=%d | loss_nats=%.6f", batch_index, batch_losses[-1])
        evaluation = evaluate_splits(model.log_prob, splits, device, epoch)
        mean_loss_bits = float(np.mean(batch_losses)) / (
            splits.sequence_length * math.log(2.0)
        )
        history.append(
            {
                "epoch": epoch,
                "mean_minibatch_nll_bits_per_site": mean_loss_bits,
                "seconds": time.perf_counter() - started,
                "evaluation": evaluation,
            }
        )
        logger.info(
            "TRAIN_EPOCH_COMPLETED | epoch=%d/%d | minibatch_nll_bits_per_site=%.6f | train_nll_bits_per_site=%.6f | seconds=%.2f",
            epoch,
            epochs,
            mean_loss_bits,
            evaluation["train"]["nll_bits_per_site"],
            history[-1]["seconds"],
        )
    return history


def fit_kgram(model: KGramModel, splits: SequenceSplits, cfg: dict) -> list[dict]:
    """Fit the B5 counting model (its single natural iteration) and evaluate it."""

    def log_prob(sequences: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(model.log_prob(sequences.cpu().numpy()))

    logger.info(
        "TRAINING_STARTED | model=kgram_counting | order=%d | smoothing=%g | iterations=1",
        model.order,
        model.smoothing,
    )
    started = time.perf_counter()
    model.fit(splits.train)
    evaluation = evaluate_splits(log_prob, splits, cfg["device"], 1)
    logger.info(
        "TRAIN_ITERATION_COMPLETED | iteration=1/1 | train_nll_bits_per_site=%.8f | seconds=%.3f",
        evaluation["train"]["nll_bits_per_site"],
        time.perf_counter() - started,
    )
    return [
        {
            "iteration": 1,
            "seconds": time.perf_counter() - started,
            "evaluation": evaluation,
        }
    ]


def select_checkpoint(history: list[dict], rule: str) -> dict:
    """Return the checkpoint entry of a training history under a declared rule.

    Parameters
    ----------
    history : list of dict
        Per-epoch / per-iteration entries with an ``evaluation`` mapping.
    rule : str
        ``"final"``: the last entry (paper protocol, Phases 3-4).
        ``"best_validation"``: the entry with the lowest validation NLL, ties
        to the earliest (Phase 5 arm B); requires a validation split.

    Returns
    -------
    dict
        ``{"rule", "index", "step", "evaluation"}``.

    Raises
    ------
    ValueError
        If the rule is unknown or ``best_validation`` has no validation split.
    """
    if rule == "final":
        index = len(history) - 1
    elif rule == "best_validation":
        if "validation" not in history[-1]["evaluation"]:
            raise ValueError("checkpoint 'best_validation' needs a validation split")
        values = [
            entry["evaluation"]["validation"]["nll_bits_per_site"] for entry in history
        ]
        index = int(np.argmin(values))  # first minimum = earliest on ties
    else:
        raise ValueError(f"Unknown checkpoint rule {rule!r}")
    entry = history[index]
    return {
        "rule": rule,
        "index": index,
        "step": entry.get("epoch", entry.get("iteration")),
        "evaluation": entry["evaluation"],
    }
