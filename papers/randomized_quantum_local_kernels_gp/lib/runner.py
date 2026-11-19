"""Runtime entry point for the randomized quantum local kernels GP reproduction.

Task ``gp_mkl`` runs the paper's pipeline end to end (Sec. III-V):

1. generate the synthetic GP data of every query kernel (``lib.data``);
2. draw R circuit parameter sets from a scrambled Sobol sequence and simulate
   the ring circuits on all inputs (``lib.circuits``);
3. compute the m = 6n local Pauli features exactly or under a finite-shot
   model (``lib.features``);
4. wrap each feature map in the cosine kernel of Eq. 21 (``lib.kernels``);
5. fit the softmax mixture weights and σ² by maximizing the GP log marginal
   likelihood, separately for every query kernel (``lib.mkl``);
6. evaluate test alignment and predictive metrics (``lib.evaluation``).

Artifacts written to the run directory (the reproduction contract):

- ``run.log``, ``run_status.json``, ``config_snapshot.json`` (shared runtime);
- ``metrics.json``: per query kernel, the learned mixture and the two
  controls (uniform weights, best single base kernel), plus feature-error
  statistics;
- ``predictions.npz``: test indices, posterior means, predictive standard
  deviations, learned weights and test Gram matrices;
- ``features.npz`` (when ``evaluation.save_features``): exact and used
  features, shape (R, M, m);
- ``fig3_datasets.png``, ``fig4_test_predictions.png``,
  ``fig5_gram_heatmap_<kernel>.png`` (when ``evaluation.figures``).
"""

from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import numpy as np

from .circuits import RingArchitecture, sample_parameter_draws, simulate_states
from .classical_baselines import (
    fit_parametric_gp,
    parametric_gram,
    rbf_dictionary_grams,
    rff_dictionary_features,
)
from .data import make_dataset, save_dataset_cache
from .evaluation import evaluate_mixture
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .features import FEATURE_MODES, estimate_features, exact_features
from .gp import factorize
from .kernels import cosine_wrapper_gram, query_kernel_matrix
from .mkl import MKLSettings, fit_mixture, fit_mixtures
from .photonic_features import (
    PhotonicArchitecture,
    photonic_features,
    sample_photonic_draws,
)
from .plotting import plot_datasets, plot_gram_heatmaps, plot_test_predictions

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = ("gp_mkl", "gp_parametric")
# "qubit_statevector" is the paper's gate-based circuit; "merlin_photonic" is
# the Phase 4 linear-optical adaptation (lib/photonic_features.py).
FEATURE_MODES_BY_BACKEND = {
    "qubit_statevector": FEATURE_MODES,
    "merlin_photonic": ("exact", "photon_sampled"),
    # Phase 5 classical dictionaries (lib/classical_baselines.py).
    "classical_rff": ("exact",),
    "rbf_dictionary": ("exact",),
}
STATUS_SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def _check_runtime_options(cfg: dict) -> None:
    task = cfg["experiment"]["task"]
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported experiment.task={task!r}; expected one of {SUPPORTED_TASKS}"
        )
    # The pipeline is NumPy-only (16-dimensional statevectors and 375 x 375
    # Gram matrices), so there is no accelerator path to select.
    if cfg["device"] != "cpu":
        raise ValueError(
            f"This reproduction runs on NumPy/CPU only; got device={cfg['device']!r}."
        )
    dtype = cfg.get("dtype")
    dtype_label = getattr(dtype, "label", dtype)
    if dtype_label != "float64":
        raise ValueError(
            f"This reproduction uses float64 throughout; got dtype={dtype_label!r}."
        )
    backend = cfg["features"]["backend"]
    if backend not in FEATURE_MODES_BY_BACKEND:
        raise ValueError(
            f"features.backend must be one of {tuple(FEATURE_MODES_BY_BACKEND)}; got {backend!r}."
        )
    if cfg["features"]["mode"] not in FEATURE_MODES_BY_BACKEND[backend]:
        raise ValueError(
            f"features.mode for backend {backend!r} must be one of "
            f"{FEATURE_MODES_BY_BACKEND[backend]}."
        )


def _mkl_settings(mkl_cfg: dict) -> MKLSettings:
    return MKLSettings(
        gradient=mkl_cfg["gradient"],
        max_iterations=int(mkl_cfg["max_iterations"]),
        tolerance=float(mkl_cfg["tolerance"]),
        coordinate_rounds=int(mkl_cfg["coordinate_rounds"]),
        initial_noise_variance=float(mkl_cfg["initial_noise_variance"]),
        fit_noise_variance=bool(mkl_cfg["fit_noise_variance"]),
        approx_rank=mkl_cfg.get("approx_rank"),
        hutchinson_probes=mkl_cfg.get("hutchinson_probes"),
        approx_seed=mkl_cfg.get("approx_seed"),
        optimizer=mkl_cfg["optimizer"],
        noise_coupling=mkl_cfg["noise_coupling"],
    )


def _control_settings(mkl_cfg: dict) -> MKLSettings:
    """Settings for the uniform and best-single controls.

    No weight is learned: only the authors' σ² grid search runs, with the
    same rounds and initial σ² whatever optimizer the learned mixture uses,
    so control values are identical across optimizer variants.
    """
    return MKLSettings(
        gradient="exact",
        max_iterations=0,
        tolerance=float(mkl_cfg["tolerance"]),
        coordinate_rounds=int(mkl_cfg["coordinate_rounds"]),
        initial_noise_variance=float(mkl_cfg["initial_noise_variance"]),
        fit_noise_variance=bool(mkl_cfg["fit_noise_variance"]),
        optimizer="authors_protocol",
        noise_coupling="per_kernel",
    )


def _initial_status(cfg: dict, run_dir: Path) -> dict:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sweep_id": cfg["experiment"].get("sweep_id"),
        "candidate": cfg["experiment"].get("candidate", {}),
        "seed": {
            "global": cfg.get("seed"),
            "data": cfg["data"]["seed"],
            "sobol": cfg["circuit"]["sobol_seed"],
            "features": cfg["features"].get("seed"),
        },
        "repetition": cfg["experiment"].get("repetition"),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_REPO_ROOT),
        "dataset": {
            "name": "synthetic_gp_"
            + "_".join(spec["name"] for spec in cfg["data"]["query_kernels"]),
            "splits": None,
            "preprocessing": "none (inputs Unif[-1, 1], labels sampled from the query-kernel GP)",
            "subset": "full generated set",
        },
        "selection": {
            "name": "train_log_marginal_likelihood",
            "direction": "maximize",
            "split": "train",
            "checkpoint": "final optimizer state",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(run_dir / "config_snapshot.json"),
        "metrics_path": None,
        "metrics_sha256": None,
        "error": None,
    }


def _compute_features(
    cfg: dict, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Exact and used feature tensors (R, M, m), the number of 1-local features, extra metrics."""
    if cfg["features"]["backend"] == "merlin_photonic":
        return _compute_photonic_features(cfg, inputs)
    exact, used = _compute_qubit_features(cfg, inputs)
    return exact, used, 3 * int(cfg["circuit"]["num_qubits"]), {}


def _compute_photonic_features(
    cfg: dict, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    photonic_cfg = cfg["photonic"]
    feature_cfg = cfg["features"]
    arch = PhotonicArchitecture(
        num_modes=int(photonic_cfg["num_modes"]),
        num_photons=int(photonic_cfg["num_photons"]),
        num_layers=int(photonic_cfg["num_layers"]),
        input_dim=inputs.shape[1],
        initial_mixing=photonic_cfg["initial_mixing"],
    )
    detector = photonic_cfg["detector"]
    draws = sample_photonic_draws(
        arch, int(cfg["circuit"]["num_draws"]), int(cfg["circuit"]["sobol_seed"])
    )
    logger.info(
        "CIRCUITS_SAMPLED | backend=merlin_photonic | num_draws=%d | sobol_dimension=%d"
        " | num_modes=%d | num_photons=%d | num_layers=%d | input_state=%s | detector=%s",
        len(draws),
        arch.sobol_dimension,
        arch.num_modes,
        arch.num_photons,
        arch.num_layers,
        arch.input_state,
        detector,
    )
    sampled = feature_cfg["mode"] == "photon_sampled"
    if sampled and feature_cfg["seed"] is None:
        raise ValueError("features.seed is required for photon_sampled features.")
    num_shots = int(feature_cfg["num_snapshots"]) if sampled else None
    rng = np.random.default_rng(feature_cfg["seed"]) if sampled else None
    exact_all, used_all, postselection_all = [], [], []
    started = time.perf_counter()
    for index, params in enumerate(draws, start=1):
        result = photonic_features(arch, params, inputs, detector, num_shots, rng)
        exact_all.append(result["exact"])
        used_all.append(result["used"])
        if "postselection" in result:
            postselection_all.append(result["postselection"])
        logger.debug("FEATURE_DRAW_COMPLETED | draw=%d/%d", index, len(draws))
    exact_tensor, used_tensor = np.stack(exact_all), np.stack(used_all)
    seconds = time.perf_counter() - started
    hardware = {
        "simulator": "MerLin QuantumLayer analytic probabilities (CPU, float64)",
        "computation_space": "UNBUNCHED" if detector == "threshold" else "FOCK",
        "detector_model": "threshold"
        if detector == "threshold"
        else "photon-number-resolving",
        "num_photons": arch.num_photons,
        "num_modes": arch.num_modes,
        "num_layers": arch.num_layers,
        "input_state": arch.input_state,
        "initial_mixing": arch.initial_mixing,
        "encoding": "phase shifters PS(delta_lq * xi_l * x) on every mode, every layer",
        "measurement": "local parities <Pi_q>, <Pi_q Pi_q+1> from one detection basis",
        "postselection": (
            f"{arch.num_photons} distinct clicks (collision-free events)"
            if detector == "threshold"
            else "none"
        ),
        "shots_per_input_and_draw": num_shots,
        "shot_sampling": "numpy multinomial on MerLin's analytic distribution"
        if sampled
        else "none (analytic)",
        "feature_seconds": seconds,
    }
    if postselection_all:
        rates = np.stack(postselection_all)
        hardware["postselection_probability_mean"] = float(rates.mean())
        hardware["postselection_probability_min"] = float(rates.min())
    logger.info(
        "FEATURES_READY | backend=merlin_photonic | mode=%s | shape=%s | sup_norm_error=%.4f"
        " | postselection_mean=%s | seconds=%.1f",
        feature_cfg["mode"],
        used_tensor.shape,
        float(np.max(np.abs(used_tensor - exact_tensor))),
        hardware.get("postselection_probability_mean"),
        seconds,
    )
    return exact_tensor, used_tensor, arch.num_modes, {"photonic": hardware}


def _compute_qubit_features(
    cfg: dict, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    circuit_cfg = cfg["circuit"]
    feature_cfg = cfg["features"]
    arch = RingArchitecture(
        num_qubits=int(circuit_cfg["num_qubits"]),
        num_layers=int(circuit_cfg["num_layers"]),
        input_dim=inputs.shape[1],
    )
    draws = sample_parameter_draws(
        arch, int(circuit_cfg["num_draws"]), int(circuit_cfg["sobol_seed"])
    )
    logger.info(
        "CIRCUITS_SAMPLED | num_draws=%d | sobol_dimension=%d | num_qubits=%d | num_layers=%d",
        len(draws),
        arch.sobol_dimension,
        arch.num_qubits,
        arch.num_layers,
    )
    if feature_cfg["mode"] != "exact" and feature_cfg["seed"] is None:
        raise ValueError(
            f"features.seed is required for feature mode {feature_cfg['mode']!r}."
        )
    rng = np.random.default_rng(feature_cfg["seed"])
    exact_all, used_all = [], []
    started = time.perf_counter()
    for index, params in enumerate(draws, start=1):
        # circuit.input_scale multiplies x before encoding (1.0 = paper);
        # the Phase 5 bandwidth diagnostic widens the encoded frequency range.
        states = simulate_states(
            arch, params, float(circuit_cfg["input_scale"]) * inputs
        )
        exact = exact_features(states, arch.num_qubits)
        if feature_cfg["mode"] == "exact":
            used = exact
        else:
            used = estimate_features(
                states,
                arch.num_qubits,
                feature_cfg["mode"],
                feature_cfg.get("num_snapshots"),
                feature_cfg.get("num_groups"),
                rng,
            )
        exact_all.append(exact)
        used_all.append(used)
        logger.debug("FEATURE_DRAW_COMPLETED | draw=%d/%d", index, len(draws))
    exact_tensor, used_tensor = np.stack(exact_all), np.stack(used_all)
    logger.info(
        "FEATURES_READY | mode=%s | shape=%s | sup_norm_error=%.4f | seconds=%.1f",
        feature_cfg["mode"],
        used_tensor.shape,
        float(np.max(np.abs(used_tensor - exact_tensor))),
        time.perf_counter() - started,
    )
    return exact_tensor, used_tensor


def _feature_error_stats(
    exact: np.ndarray, used: np.ndarray, num_one_local: int
) -> dict:
    error = np.abs(used - exact)
    return {
        "sup_norm": float(error.max()),
        "mean_abs": float(error.mean()),
        "sup_norm_one_local": float(error[..., :num_one_local].max()),
        "sup_norm_two_local": float(error[..., num_one_local:].max()),
    }


def _evaluate_on_split(weights, noise, *, split_context: dict, grams: np.ndarray):
    return evaluate_mixture(
        grams,
        weights,
        noise,
        split_context["train_index"],
        split_context["test_index"],
        split_context["train_targets"],
        split_context["test_targets"],
        split_context["true_test_gram"],
    )


def _prepare_dataset(cfg: dict, run_dir: Path, status: dict):
    dataset = make_dataset(cfg)
    train_index, test_index = dataset.train_index, dataset.test_index
    status["dataset"]["splits"] = {
        "train": int(train_index.size),
        "test": int(test_index.size),
    }
    status["dataset"]["sha256"] = dataset.fingerprint()
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "DATASET_READY | kernels=%s | num_points=%d | train=%d | test=%d | sha256=%s",
        ",".join(dataset.kernel_names),
        dataset.inputs.shape[0],
        train_index.size,
        test_index.size,
        status["dataset"]["sha256"],
    )
    if cfg["data"].get("cache", False):
        save_dataset_cache(cfg, dataset)
    return dataset


def _compute_dictionary(cfg: dict, inputs: np.ndarray) -> dict:
    """Base Gram matrices (R, M, M) of the configured dictionary, plus feature diagnostics."""
    backend = cfg["features"]["backend"]
    if backend == "rbf_dictionary":
        classical_cfg = cfg["classical"]
        grams, lengthscales = rbf_dictionary_grams(
            inputs,
            int(cfg["circuit"]["num_draws"]),
            tuple(classical_cfg["lengthscale_range"]),
        )
        logger.info(
            "BASE_GRAMS_READY | backend=rbf_dictionary | shape=%s | lengthscales=[%.4g, %.4g]",
            grams.shape,
            lengthscales.min(),
            lengthscales.max(),
        )
        return {
            "grams": grams,
            "features": None,
            "feature_error": None,
            "extras": {
                "classical": {
                    "dictionary": "rbf",
                    "lengthscale_range": list(classical_cfg["lengthscale_range"]),
                }
            },
        }
    if backend == "classical_rff":
        classical_cfg = cfg["classical"]
        started = time.perf_counter()
        features, lengthscales = rff_dictionary_features(
            inputs,
            int(cfg["circuit"]["num_draws"]),
            int(classical_cfg["num_features"]),
            int(cfg["circuit"]["sobol_seed"]),
            tuple(classical_cfg["lengthscale_range"]),
        )
        logger.info(
            "FEATURES_READY | backend=classical_rff | shape=%s | lengthscales=[%.4g, %.4g] | seconds=%.1f",
            features.shape,
            lengthscales.min(),
            lengthscales.max(),
            time.perf_counter() - started,
        )
        exact, used, num_one_local, extras = (
            features,
            features,
            features.shape[-1] // 2,
            {
                "classical": {
                    "dictionary": "rff",
                    "num_features": int(classical_cfg["num_features"]),
                    "lengthscale_range": list(classical_cfg["lengthscale_range"]),
                }
            },
        )
    else:
        exact, used, num_one_local, extras = _compute_features(cfg, inputs)
    grams = cosine_wrapper_gram(used, used)
    logger.info("BASE_GRAMS_READY | shape=%s", grams.shape)
    return {
        "grams": grams,
        "features": (exact, used),
        "feature_error": _feature_error_stats(exact, used, num_one_local),
        "extras": extras,
    }


def _run_gp_mkl(cfg: dict, run_dir: Path, status: dict) -> None:
    dataset = _prepare_dataset(cfg, run_dir, status)
    train_index, test_index = dataset.train_index, dataset.test_index

    dictionary = _compute_dictionary(cfg, dataset.inputs)
    grams = dictionary["grams"]

    settings = _mkl_settings(cfg["mkl"])
    uniform_settings = _control_settings(cfg["mkl"])
    train_grams = grams[:, train_index][:, :, train_index]
    query_specs = cfg["data"]["query_kernels"]

    logger.info(
        "TRAINING_STARTED | optimizer=%s | noise_coupling=%s | gradient=%s | kernels=%d"
        " | num_base_kernels=%d | max_iterations=%d | coordinate_rounds=%d",
        settings.optimizer,
        settings.noise_coupling,
        settings.gradient,
        len(query_specs),
        grams.shape[0],
        settings.max_iterations,
        settings.coordinate_rounds,
    )
    fits, controls = {}, {}
    names = [spec["name"] for spec in query_specs]
    started = time.perf_counter()
    fitted = fit_mixtures(
        train_grams,
        [dataset.labels[index, train_index] for index in range(len(query_specs))],
        settings,
        names,
    )
    elapsed = time.perf_counter() - started
    for name, result in zip(names, fitted):
        # With shared-σ² coupling the GPs are fitted jointly, so only the
        # total time is meaningful; it is split evenly for reporting.
        result.seconds = elapsed / len(names)
        fits[name] = result
        logger.info(
            "TRAINING_KERNEL_COMPLETED | kernel=%s | iterations=%d | log_marginal=%.6f"
            " | noise_variance=%.3e | seconds=%.1f",
            name,
            result.iterations,
            result.log_marginal,
            result.noise_variance,
            result.seconds,
        )
    for kernel_index, spec in enumerate(query_specs):
        name = spec["name"]
        targets = dataset.labels[kernel_index, train_index]
        if cfg["evaluation"]["controls"]:
            uniform = fit_mixture(
                train_grams, targets, uniform_settings, label=f"uniform:{name}"
            )
            single_scores = [
                factorize(
                    train_grams[r], fits[name].noise_variance, targets
                ).log_marginal
                for r in range(train_grams.shape[0])
            ]
            best_draw = int(np.argmax(single_scores))
            single = fit_mixture(
                train_grams[best_draw : best_draw + 1],
                targets,
                uniform_settings,
                label=f"best_single:{name}",
            )
            controls[name] = {"uniform": uniform, "best_single": (best_draw, single)}

    logger.info("EVALUATION_STARTED | split=test")
    metrics: dict = {
        "task": "gp_mkl",
        "experiment": cfg["experiment"].get("name"),
        "feature_backend": cfg["features"]["backend"],
        "feature_mode": cfg["features"]["mode"],
        "feature_error": dictionary["feature_error"],
        **dictionary["extras"],
        "num_base_kernels": int(grams.shape[0]),
        "num_features": (
            int(dictionary["features"][1].shape[-1]) if dictionary["features"] else None
        ),
        "kernels": {},
    }
    test_inputs = dataset.inputs[test_index]
    means, stds, learned_test_grams, true_test_grams = [], [], [], []
    for kernel_index, spec in enumerate(query_specs):
        name = spec["name"]
        true_test_gram = query_kernel_matrix(
            test_inputs, test_inputs, spec["type"], spec["params"]
        )
        train_targets = dataset.labels[kernel_index, train_index]
        test_targets = dataset.labels[kernel_index, test_index]

        split_context = {
            "train_index": train_index,
            "test_index": test_index,
            "train_targets": train_targets,
            "test_targets": test_targets,
            "true_test_gram": true_test_gram,
        }
        evaluate = partial(_evaluate_on_split, split_context=split_context, grams=grams)

        learned = evaluate(fits[name].weights, fits[name].noise_variance)
        entry = {
            "learned_mixture": {
                **learned.metrics,
                "iterations": fits[name].iterations,
                "fit_seconds": fits[name].seconds,
            }
        }
        if name in controls:
            uniform = controls[name]["uniform"]
            best_draw, single = controls[name]["best_single"]
            entry["uniform_mixture"] = evaluate(
                uniform.weights, uniform.noise_variance
            ).metrics
            entry["best_single_kernel"] = {
                **evaluate(
                    single.weights,
                    single.noise_variance,
                    grams=grams[best_draw : best_draw + 1],
                ).metrics,
                "draw_index": best_draw,
            }
        metrics["kernels"][name] = entry
        logger.info(
            "EVALUATION_COMPLETED | split=test | kernel=%s | alignment_centered=%.5f"
            " | alignment_uncentered=%.5f | rmse=%.5f | avg_nlpd=%.4f | coverage_2sigma=%.3f",
            name,
            learned.metrics["alignment_centered"],
            learned.metrics["alignment_uncentered"],
            learned.metrics["rmse"],
            learned.metrics["avg_nlpd"],
            learned.metrics["coverage_2sigma"],
        )
        means.append(learned.mean)
        stds.append(learned.predictive_std)
        learned_test_grams.append(learned.test_gram)
        true_test_grams.append(true_test_gram)

    np.savez_compressed(
        run_dir / "predictions.npz",
        kernel_names=np.array(dataset.kernel_names),
        test_index=test_index,
        test_inputs=test_inputs,
        test_targets=dataset.labels[:, test_index],
        means=np.stack(means),
        predictive_std=np.stack(stds),
        weights=np.stack([fits[spec["name"]].weights for spec in query_specs]),
        noise_variances=np.array(
            [fits[spec["name"]].noise_variance for spec in query_specs]
        ),
        learned_test_grams=np.stack(learned_test_grams),
        true_test_grams=np.stack(true_test_grams),
    )
    logger.info("ARTIFACT_WRITTEN | path=predictions.npz")
    if cfg["evaluation"]["save_features"] and dictionary["features"] is not None:
        np.savez_compressed(
            run_dir / "features.npz",
            exact=dictionary["features"][0],
            used=dictionary["features"][1],
        )
        logger.info("ARTIFACT_WRITTEN | path=features.npz")
    if cfg["evaluation"]["figures"]:
        plot_datasets(
            dataset.inputs,
            dataset.labels,
            dataset.kernel_names,
            run_dir / "fig3_datasets.png",
        )
        plot_test_predictions(
            test_inputs,
            dataset.labels[:, test_index],
            np.stack(means),
            np.stack(stds),
            dataset.kernel_names,
            run_dir / "fig4_test_predictions.png",
        )
        for name, true_gram, learned_gram in zip(
            dataset.kernel_names, true_test_grams, learned_test_grams
        ):
            plot_gram_heatmaps(
                true_gram, learned_gram, name, run_dir / f"fig5_gram_heatmap_{name}.png"
            )
        logger.info(
            "ARTIFACT_WRITTEN | figures=fig3_datasets.png,fig4_test_predictions.png,fig5_gram_heatmap_*.png"
        )

    write_json(run_dir / "metrics.json", metrics)
    status["metrics_path"] = "metrics.json"
    status["metrics_sha256"] = sha256(run_dir / "metrics.json")
    logger.info("METRICS_WRITTEN | path=metrics.json")


def _run_gp_parametric(cfg: dict, run_dir: Path, status: dict) -> None:
    """Phase 5 baselines B1/B2: ML-II parametric GPs evaluated like the mixtures."""
    dataset = _prepare_dataset(cfg, run_dir, status)
    train_index, test_index = dataset.train_index, dataset.test_index
    baseline_cfg = cfg["parametric"]
    restarts = int(baseline_cfg["restarts"])
    seed = int(cfg["circuit"]["sobol_seed"])
    query_specs = cfg["data"]["query_kernels"]
    families = {
        spec["name"]: (
            spec["type"] if baseline_cfg["family"] == "true" else baseline_cfg["family"]
        )
        for spec in query_specs
    }
    logger.info(
        "TRAINING_STARTED | optimizer=ml2_lbfgsb | family=%s | restarts=%d | kernels=%d",
        baseline_cfg["family"],
        restarts,
        len(query_specs),
    )
    fits = {}
    for kernel_index, spec in enumerate(query_specs):
        started = time.perf_counter()
        fits[spec["name"]] = fit_parametric_gp(
            families[spec["name"]],
            dataset.inputs[train_index],
            dataset.labels[kernel_index, train_index],
            restarts,
            seed + kernel_index,
            spec["name"],
        )
        fits[spec["name"]].seconds = time.perf_counter() - started
        logger.info(
            "TRAINING_KERNEL_COMPLETED | kernel=%s | family=%s | log_marginal=%.6f"
            " | noise_variance=%.3e | params=%s | seconds=%.1f",
            spec["name"],
            families[spec["name"]],
            fits[spec["name"]].log_marginal,
            fits[spec["name"]].noise_variance,
            fits[spec["name"]].params,
            fits[spec["name"]].seconds,
        )

    logger.info("EVALUATION_STARTED | split=test")
    metrics: dict = {
        "task": "gp_parametric",
        "experiment": cfg["experiment"].get("name"),
        "parametric_family": baseline_cfg["family"],
        "kernels": {},
    }
    inputs = dataset.inputs
    for kernel_index, spec in enumerate(query_specs):
        name = spec["name"]
        fit = fits[name]
        test_inputs = inputs[test_index]
        true_test_gram = query_kernel_matrix(
            test_inputs, test_inputs, spec["type"], spec["params"]
        )
        full_gram = parametric_gram(fit.family, fit.params, inputs, inputs)
        evaluation = evaluate_mixture(
            full_gram[None],
            np.ones(1),
            fit.noise_variance,
            train_index,
            test_index,
            dataset.labels[kernel_index, train_index],
            dataset.labels[kernel_index, test_index],
            true_test_gram,
        )
        evaluation.metrics.update(
            family=fit.family,
            fitted_params=fit.params,
            restarts=fit.restarts,
            fit_seconds=fit.seconds,
        )
        # The single-kernel "mixture" has trivial weight diagnostics.
        for key in ("effective_num_kernels", "max_weight"):
            evaluation.metrics.pop(key)
        metrics["kernels"][name] = {"parametric_gp": evaluation.metrics}
        logger.info(
            "EVALUATION_COMPLETED | split=test | kernel=%s | alignment_centered=%.5f"
            " | alignment_uncentered=%.5f | rmse=%.5f | avg_nlpd=%.4f | coverage_2sigma=%.3f",
            name,
            evaluation.metrics["alignment_centered"],
            evaluation.metrics["alignment_uncentered"],
            evaluation.metrics["rmse"],
            evaluation.metrics["avg_nlpd"],
            evaluation.metrics["coverage_2sigma"],
        )
    write_json(run_dir / "metrics.json", metrics)
    status["metrics_path"] = "metrics.json"
    status["metrics_sha256"] = sha256(run_dir / "metrics.json")
    logger.info("METRICS_WRITTEN | path=metrics.json")


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the configured experiment task.

    Parameters
    ----------
    cfg : dict
        Resolved configuration from the shared runtime.
    run_dir : pathlib.Path
        Timestamped output directory for this run. The shared runtime has
        already written ``config_snapshot.json`` there.

    Raises
    ------
    ValueError
        If the task, device or dtype is not supported.
    """
    run_dir = Path(run_dir)
    level_name = str(cfg.get("logging", {}).get("level", "info")).lower()
    configure_logging(
        run_dir / "run.log", level=_LOG_LEVELS.get(level_name, logging.INFO)
    )
    _check_runtime_options(cfg)
    status = _initial_status(cfg, run_dir)
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | task=%s | experiment=%s | seed=%s",
        status["run_id"],
        cfg["experiment"]["task"],
        cfg["experiment"].get("name"),
        status["seed"],
    )
    try:
        if cfg["experiment"]["task"] == "gp_mkl":
            _run_gp_mkl(cfg, run_dir, status)
        else:
            _run_gp_parametric(cfg, run_dir, status)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", status["run_id"])
        raise
    status.update(status="COMPLETED", completed_at=utc_now())
    write_json(run_dir / "run_status.json", status)
    logger.info("RUN_COMPLETED | run_id=%s", status["run_id"])
