from __future__ import annotations

import pytest
from common import PROJECT_DIR  # noqa: F401  (puts the repo root on sys.path)
from quantum_correlations_generative_models.lib.data import (
    PAPER_SHAPES,
    build_dataloader,
    load_splits,
)

DATASETS = [
    ("spect_heart", None, 80, 187),
    ("spect_heart", 0.25, 201, 66),
    ("promoter_gene_sequences", 0.25, 80, 26),
    ("biofam", 0.25, 1500, 500),
]


def _cfg(name, test_fraction, validation_fraction=0.0):
    return {
        "seed": 0,
        "device": "cpu",
        "dataset": {
            "name": name,
            "split_seed": 0,
            "test_fraction": test_fraction,
            "validation_fraction": validation_fraction,
            "batch_size": 8,
        },
    }


@pytest.mark.parametrize("name,test_fraction,num_train,num_test", DATASETS)
def test_one_batch_has_paper_shape(name, test_fraction, num_train, num_test):
    cfg = _cfg(name, test_fraction)
    splits = load_splits(cfg)
    batch = next(iter(build_dataloader(cfg, "train", splits)))
    length, alphabet = PAPER_SHAPES[name]
    assert batch[0].shape == (8, length)
    assert int(batch[0].max()) < alphabet
    assert (len(splits.train), len(splits.test)) == (num_train, num_test)


def test_validation_carve_out_keeps_test_set():
    plain = load_splits(_cfg("biofam", 0.25))
    carved = load_splits(_cfg("biofam", 0.25, validation_fraction=0.2))
    assert (plain.test == carved.test).all()
    assert len(carved.validation) == 300 and len(carved.train) == 1200


def test_pooled_spect_split_partitions_both_files():
    standard = load_splits(_cfg("spect_heart", None))
    pooled = load_splits(_cfg("spect_heart", 0.25))

    def as_sorted_rows(*arrays):
        return sorted(tuple(row) for array in arrays for row in array.tolist())

    assert as_sorted_rows(pooled.train, pooled.test) == as_sorted_rows(
        standard.train, standard.test
    )
