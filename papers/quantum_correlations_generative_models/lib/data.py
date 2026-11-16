"""Sequence datasets of Sec. V C (PRX 12, 021037, p13).

Each dataset is a matrix of categorical sequences, shape ``(N, n)`` with
integer values in ``0..M-1``:

- ``biofam``: n = 16 yearly family states (ages 15-30), M = 8. Source: CRAN
  TraMineR 2.2-14, converted once to ``biofam/biofam.csv`` (columns a15..a30).
- ``promoter_gene_sequences``: n = 57 nucleotides, M = 4 (a, c, g, t -> 0..3).
  Source: UCI ``promoters.data``; the class label and instance name are
  dropped because the paper models the sequence only (n = 57).
- ``spect_heart``: n = 23 binary variables (diagnosis + F1..F22, file order),
  M = 2. Source: UCI ``SPECT.train`` (80 rows) and ``SPECT.test`` (187 rows).
  With ``test_fraction = null`` the standard files are the split, as the
  paper's text states (p13). With a numeric ``test_fraction`` the 267 rows are
  pooled and split like the other datasets: the paper's own numbers (the
  h = 1 point of Fig. 7 and the Fig. 8 threshold, which needs N_train = 200)
  match that split instead (LOG.md, R6).

The paper withholds "a quarter" of biofam and Promoters for testing without
naming it (LOG.md, R6); here the test set is the first ``floor(N *
test_fraction)`` rows of a permutation drawn with ``split_seed``. An optional
validation set is carved from the remaining training rows with the same
permutation stream, so the test set never changes with
``validation_fraction``.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PAPER_DATA_DIRNAME = "quantum_correlations_generative_models"
NUCLEOTIDE_CODES = {"a": 0, "c": 1, "g": 2, "t": 3}
BIOFAM_COLUMNS = [f"a{age}" for age in range(15, 31)]

# (sequence length n, alphabet size M) stated in Sec. V C, p13.
PAPER_SHAPES = {
    "biofam": (16, 8),
    "promoter_gene_sequences": (57, 4),
    "spect_heart": (23, 2),
}


@dataclass(frozen=True)
class SequenceSplits:
    """Integer-coded sequences split into train, validation and test.

    Attributes
    ----------
    name : str
        Dataset name.
    alphabet_size : int
        Number of categories M per variable.
    train, validation, test : numpy.ndarray
        Integer arrays of shape ``(N_split, n)``. ``validation`` may have zero
        rows.
    preprocessing : str
        Human-readable description of the encoding and split.
    """

    name: str
    alphabet_size: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    preprocessing: str

    @property
    def sequence_length(self) -> int:
        return int(self.train.shape[1])

    def split(self, split: str) -> np.ndarray:
        """Return the array of one split ("train", "validation" or "test")."""
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        return getattr(self, split)


def paper_data_dir(cfg: dict) -> Path:
    """Return ``<data_root>/quantum_correlations_generative_models``.

    ``cfg["data_root"]`` is set by the shared runtime from ``--data-root``, the
    ``DATA_DIR`` environment variable or ``<repo>/data``.
    """
    from runtime_lib.data_paths import resolve_data_root

    root = resolve_data_root(cfg.get("data_root"))
    directory = root / PAPER_DATA_DIRNAME
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing data directory {directory}; see LOG.md Data Acquisition Log "
            "for the re-acquisition commands."
        )
    return directory


def _read_biofam(directory: Path) -> np.ndarray:
    path = directory / "biofam" / "biofam.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.array(
        [[int(float(row[column])) for column in BIOFAM_COLUMNS] for row in rows],
        dtype=np.int64,
    )


def _read_promoters(directory: Path) -> np.ndarray:
    path = directory / "promoter_gene_sequences" / "raw" / "promoters.data"
    sequences = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _label, _name, sequence = line.split(",", 2)
        sequences.append([NUCLEOTIDE_CODES[base] for base in sequence.strip()])
    return np.array(sequences, dtype=np.int64)


def _read_spect_file(path: Path) -> np.ndarray:
    rows = [
        [int(value) for value in line.split(",")]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.array(rows, dtype=np.int64)


def _permuted_split(
    data: np.ndarray, fraction: float, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (held_out, remaining) with ``floor(N * fraction)`` held-out rows."""
    order = generator.permutation(len(data))
    num_held_out = int(math.floor(len(data) * fraction))
    return data[order[:num_held_out]], data[order[num_held_out:]]


def load_splits(cfg: dict) -> SequenceSplits:
    """Load and split the dataset named by ``cfg["dataset"]["name"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with ``dataset.name``, ``dataset.split_seed``,
        ``dataset.test_fraction`` (``None`` for SPECT Heart selects its standard
        test file; a number pools the two files) and
        ``dataset.validation_fraction``.

    Returns
    -------
    SequenceSplits
        Integer-coded splits.

    Raises
    ------
    ValueError
        If the dataset name is unknown, a fraction is inconsistent with the
        dataset, or the loaded shape disagrees with the paper's (n, M).
    """
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"]
    if name not in PAPER_SHAPES:
        raise ValueError(
            f"Unknown dataset {name!r}; expected one of {sorted(PAPER_SHAPES)}"
        )
    test_fraction = dataset_cfg["test_fraction"]
    validation_fraction = float(dataset_cfg["validation_fraction"])
    generator = np.random.default_rng(int(dataset_cfg["split_seed"]))
    directory = paper_data_dir(cfg)

    if name == "spect_heart" and test_fraction is None:
        raw = directory / "spect_heart" / "raw"
        train_pool = _read_spect_file(raw / "SPECT.train")
        test = _read_spect_file(raw / "SPECT.test")
        split_text = "standard SPECT.train / SPECT.test files"
    elif name == "spect_heart":
        raw = directory / "spect_heart" / "raw"
        pooled = np.concatenate(
            [
                _read_spect_file(raw / "SPECT.train"),
                _read_spect_file(raw / "SPECT.test"),
            ]
        )
        test, train_pool = _permuted_split(pooled, float(test_fraction), generator)
        split_text = (
            f"SPECT.train + SPECT.test pooled ({len(pooled)} rows); test = first "
            f"floor(N*{test_fraction}) rows of a permutation with "
            f"split_seed={dataset_cfg['split_seed']}"
        )
    else:
        if test_fraction is None:
            raise ValueError(f"{name} needs dataset.test_fraction (paper: 0.25)")
        full = (
            _read_biofam(directory) if name == "biofam" else _read_promoters(directory)
        )
        test, train_pool = _permuted_split(full, float(test_fraction), generator)
        split_text = (
            f"test = first floor(N*{test_fraction}) rows of a permutation with "
            f"split_seed={dataset_cfg['split_seed']}"
        )

    validation, train = _permuted_split(train_pool, validation_fraction, generator)
    expected_length, alphabet_size = PAPER_SHAPES[name]
    for split_name, array in (("train", train), ("test", test)):
        if array.shape[1] != expected_length:
            raise ValueError(
                f"{name} {split_name} has n={array.shape[1]}, paper states {expected_length}"
            )
        if array.min() < 0 or array.max() >= alphabet_size:
            raise ValueError(
                f"{name} {split_name} values outside 0..{alphabet_size - 1}"
            )

    preprocessing = (
        f"integer coding 0..{alphabet_size - 1}; {split_text}; validation = "
        f"floor(N_train_pool*{validation_fraction}) rows carved from training"
    )
    return SequenceSplits(name, alphabet_size, train, validation, test, preprocessing)


def build_dataloader(
    cfg: dict, split: str, splits: SequenceSplits | None = None
) -> DataLoader:
    """Return a DataLoader of ``(sequences,)`` batches for one split.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (``dataset`` section, ``seed``, ``device``).
    split : str
        "train", "validation" or "test". Only the training split is shuffled.
    splits : SequenceSplits | None
        Already loaded splits; loaded from ``cfg`` when omitted. Default value
        is None.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches are 1-tuples holding a ``torch.long`` tensor of shape
        ``(batch, n)`` on ``cfg["device"]``.
    """
    splits = splits if splits is not None else load_splits(cfg)
    sequences = torch.as_tensor(
        splits.split(split), dtype=torch.long, device=cfg["device"]
    )
    shuffle = split == "train"
    generator = torch.Generator().manual_seed(int(cfg["seed"])) if shuffle else None
    return DataLoader(
        TensorDataset(sequences),
        batch_size=int(cfg["dataset"]["batch_size"]),
        shuffle=shuffle,
        generator=generator,
    )


def dataset_identity(splits: SequenceSplits) -> dict:
    """Return the ``dataset`` block of ``run_status.json``."""
    return {
        "name": splits.name,
        "splits": {
            "train": int(len(splits.train)),
            "validation": int(len(splits.validation)),
            "test": int(len(splits.test)),
        },
        "preprocessing": splits.preprocessing,
        "subset": "full dataset",
    }
