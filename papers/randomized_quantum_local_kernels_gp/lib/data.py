"""Synthetic GP regression data (Sec. V-A/B) behind one loader interface.

The paper's data are procedural: M inputs drawn uniformly on [-1, 1] and, for
every query kernel κ*, labels y ~ N(0, K* + σ² I) on the same inputs. The RNG
sequence follows the authors' generator specification (reimplemented, not
copied): ``default_rng(seed)`` draws X, then one int64 child seed per query
kernel from the same generator, and each label vector is drawn with
``default_rng(child).multivariate_normal``. The first ⌊train_fraction·M⌋
points are the training split and the rest the test split, as in the
authors' notebook (inputs are i.i.d., so a prefix split is a random split).

``multivariate_normal`` factorizes the covariance with an SVD whose output can
differ across NumPy/LAPACK builds, so bit-identical labels with the authors'
run are not guaranteed.

No external data are read. When ``data.cache`` is true, the generated arrays
are written under ``<data_root>/randomized_quantum_local_kernels_gp/synthetic_gp/``
where ``data_root`` comes from ``--data-root``, ``DATA_DIR`` or the repository
``data/`` folder (resolved by the shared runtime).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kernels import query_kernel_matrix

logger = logging.getLogger(__name__)

PAPER_DATA_NAME = "randomized_quantum_local_kernels_gp"
SPLITS = ("train", "test")


@dataclass(frozen=True)
class GPDataset:
    """Inputs and labels of all query-kernel GPs.

    Attributes
    ----------
    inputs : np.ndarray
        Inputs of shape (M, ζ).
    labels : np.ndarray
        Noisy labels of shape (num_kernels, M), one row per query kernel.
    kernel_names : tuple of str
        Query-kernel names in label-row order.
    train_index : np.ndarray
        Indices of the training split.
    test_index : np.ndarray
        Indices of the test split.
    """

    inputs: np.ndarray
    labels: np.ndarray
    kernel_names: tuple[str, ...]
    train_index: np.ndarray
    test_index: np.ndarray

    def split_index(self, split: str) -> np.ndarray:
        """Indices of ``split`` ("train" or "test")."""
        if split == "train":
            return self.train_index
        if split == "test":
            return self.test_index
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")

    def fingerprint(self) -> str:
        """SHA-256 of the inputs and labels, for run provenance."""
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.inputs).tobytes())
        digest.update(np.ascontiguousarray(self.labels).tobytes())
        return digest.hexdigest()


def _data_config(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    required = (
        "num_points",
        "input_dim",
        "noise_variance",
        "train_fraction",
        "seed",
        "query_kernels",
    )
    missing = [key for key in required if key not in data_cfg]
    if missing:
        raise KeyError(f"data config is missing keys: {missing}")
    if data_cfg["input_dim"] != 1:
        raise ValueError(
            "The query kernels are defined for scalar inputs (data.input_dim = 1)."
        )
    return data_cfg


def make_dataset(cfg: dict) -> GPDataset:
    """Generate the synthetic GP dataset described by ``cfg["data"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with a ``data`` block.

    Returns
    -------
    GPDataset
        Inputs, labels and the train/test split.
    """
    data_cfg = _data_config(cfg)
    num_points = int(data_cfg["num_points"])
    noise_variance = float(data_cfg["noise_variance"])
    rng = np.random.default_rng(int(data_cfg["seed"]))
    inputs = rng.uniform(-1.0, 1.0, size=(num_points, int(data_cfg["input_dim"])))

    query_kernels = data_cfg["query_kernels"]
    child_seeds = rng.integers(0, 2**63 - 1, size=len(query_kernels), dtype=np.int64)
    labels = []
    for spec, child_seed in zip(query_kernels, child_seeds):
        gram = query_kernel_matrix(inputs, inputs, spec["type"], spec["params"])
        # The authors clip to [0, 1], symmetrize and set a unit diagonal; for
        # these kernels this only removes rounding error.
        gram = np.clip(gram, 0.0, 1.0)
        gram = 0.5 * (gram + gram.T)
        np.fill_diagonal(gram, 1.0)
        covariance = gram + noise_variance * np.eye(num_points)
        labels.append(
            np.random.default_rng(int(child_seed)).multivariate_normal(
                np.zeros(num_points), covariance
            )
        )

    num_train = int(np.floor(float(data_cfg["train_fraction"]) * num_points))
    if not 1 <= num_train < num_points:
        raise ValueError(
            f"train_fraction gives {num_train} training points out of {num_points}."
        )
    return GPDataset(
        inputs=inputs,
        labels=np.asarray(labels),
        kernel_names=tuple(spec["name"] for spec in query_kernels),
        train_index=np.arange(num_train),
        test_index=np.arange(num_train, num_points),
    )


def build_dataloader(cfg: dict, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Full-batch loader for one split.

    Exact GP inference uses the whole split at once, so the loader yields a
    single batch ``(inputs, labels)`` with shapes (M_split, ζ) and
    (num_kernels, M_split).

    Parameters
    ----------
    cfg : dict
        Resolved configuration with a ``data`` block.
    split : str
        "train" or "test".

    Returns
    -------
    list of tuple
        One batch.
    """
    dataset = make_dataset(cfg)
    index = dataset.split_index(split)
    return [(dataset.inputs[index], dataset.labels[:, index])]


def dataset_cache_path(cfg: dict) -> Path:
    """Cache file for the generated arrays under the shared data root."""
    data_root = cfg.get("data_root")
    if not data_root:
        raise KeyError("cfg['data_root'] is required to cache the dataset.")
    key = hashlib.sha256(json.dumps(cfg["data"], sort_keys=True).encode()).hexdigest()[
        :12
    ]
    return Path(data_root) / PAPER_DATA_NAME / "synthetic_gp" / f"gp_data_{key}.npz"


def save_dataset_cache(cfg: dict, dataset: GPDataset) -> Path:
    """Write the generated dataset to the shared data root and return its path."""
    path = dataset_cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        inputs=dataset.inputs,
        labels=dataset.labels,
        kernel_names=np.array(dataset.kernel_names),
        train_index=dataset.train_index,
        test_index=dataset.test_index,
        data_config=json.dumps(cfg["data"], sort_keys=True),
    )
    logger.info("Cached generated dataset at %s", path)
    return path
