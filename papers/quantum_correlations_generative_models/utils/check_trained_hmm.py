"""Known-answer check of a trained HMM run: normalisation and independent likelihood.

Gap diagnosis (LOG.md, Phase 3). Loads ``model_state.pt`` and
``config_snapshot.json`` of one HMM run and checks that its reported
likelihood is a real likelihood:

1. the probabilities of all M^n sequences sum to 1 (enumerated in chunks;
   only when M^n <= 2^24);
2. the same parameters embedded in the basis-enhanced model
   (``BasisEnhancedTwoGram.from_hmm``) give the same train log-likelihood,
   through an independent contraction;
3. the recomputed train NLL equals the value in ``metrics.json``.

Usage::

    python3 utils/check_trained_hmm.py <RUN_DIR> --out <JSON_PATH>
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from quantum_correlations_generative_models.lib.basis_enhanced import (  # noqa: E402
    BasisEnhancedTwoGram,
)
from quantum_correlations_generative_models.lib.data import load_splits  # noqa: E402
from quantum_correlations_generative_models.lib.hmm import HiddenMarkovModel  # noqa: E402

MAX_ENUMERATED = 2**24
CHUNK = 2**18


def total_probability(model: HiddenMarkovModel, length: int, alphabet: int) -> float:
    """Sum p(x) over every sequence of the given length."""
    total = 0.0
    all_rows = itertools.product(range(alphabet), repeat=length)
    while True:
        chunk = list(itertools.islice(all_rows, CHUNK))
        if not chunk:
            return total
        sequences = torch.tensor(chunk, dtype=torch.long)
        total += float(torch.exp(model.log_prob(sequences)).sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads((args.run_dir / "config_snapshot.json").read_text())
    if cfg["model"]["kind"] != "hmm_baum_welch":
        raise ValueError("check_trained_hmm.py expects an hmm_baum_welch run")
    cfg["data_root"] = cfg.get("data_root")
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    state = torch.load(args.run_dir / "model_state.pt")
    model = HiddenMarkovModel(state["initial"], state["transition"], state["emission"])
    splits = load_splits(cfg)
    length, alphabet = splits.sequence_length, splits.alphabet_size
    train = torch.as_tensor(splits.train, dtype=torch.long)

    with torch.no_grad():
        hmm_log_probs = model.log_prob(train)
        embedded = BasisEnhancedTwoGram(
            model.transition.shape[0],
            alphabet,
            torch.Generator().manual_seed(0),
            torch.float64,
        )
        embedded.from_hmm(model.initial, model.transition, model.emission)
        embedded_log_probs = embedded.log_prob(train)
    recomputed_nll = float(-hmm_log_probs.mean()) / (length * math.log(2.0))
    result = {
        "run_dir": str(args.run_dir),
        "dataset": splits.name,
        "bond_dim": int(model.transition.shape[0]),
        "recomputed_train_nll_bits_per_site": recomputed_nll,
        "reported_train_nll_bits_per_site": metrics["final"]["train"][
            "nll_bits_per_site"
        ],
        "max_abs_log_prob_difference_hmm_vs_embedded": float(
            (hmm_log_probs - embedded_log_probs).abs().max()
        ),
    }
    if alphabet**length <= MAX_ENUMERATED:
        result["total_probability_over_all_sequences"] = total_probability(
            model, length, alphabet
        )
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
