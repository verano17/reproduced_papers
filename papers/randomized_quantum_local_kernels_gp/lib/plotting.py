"""Figures matching the paper's Fig. 3 (data), Fig. 4 (predictions) and Fig. 5 (Gram heatmaps)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _grid(count: int) -> tuple[plt.Figure, np.ndarray]:
    columns = min(2, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.0 * rows), squeeze=False
    )
    for extra in axes.flat[count:]:
        extra.axis("off")
    return figure, axes.flat


def plot_datasets(
    inputs: np.ndarray, labels: np.ndarray, names: tuple[str, ...], path: Path
) -> None:
    """Scatter of y against x for every query kernel (Fig. 3)."""
    figure, axes = _grid(len(names))
    for axis, name, row in zip(axes, names, labels):
        axis.scatter(inputs[:, 0], row, s=4)
        axis.set_title(name)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_test_predictions(
    test_inputs: np.ndarray,
    test_labels: np.ndarray,
    means: np.ndarray,
    predictive_std: np.ndarray,
    names: tuple[str, ...],
    path: Path,
) -> None:
    """Test observations, predictive mean and ±2σ band of the noisy targets (Fig. 4)."""
    order = np.argsort(test_inputs[:, 0])
    x_sorted = test_inputs[order, 0]
    figure, axes = _grid(len(names))
    for axis, name, labels, mean, std in zip(
        axes, names, test_labels, means, predictive_std
    ):
        axis.scatter(x_sorted, labels[order], s=6, label="test obs (true)")
        axis.plot(x_sorted, mean[order], color="tab:orange", label="predictive mean")
        axis.fill_between(
            x_sorted,
            mean[order] - 2.0 * std[order],
            mean[order] + 2.0 * std[order],
            color="0.7",
            alpha=0.6,
            label="±2σ",
        )
        axis.set_title(name)
        axis.set_xlabel("x (test points)")
        axis.set_ylabel("y")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _min_max(gram: np.ndarray) -> np.ndarray:
    span = gram.max() - gram.min()
    if span == 0.0:
        raise ValueError("Cannot min-max scale a constant Gram matrix.")
    return (gram - gram.min()) / span


def plot_gram_heatmaps(
    true_gram: np.ndarray, learned_gram: np.ndarray, name: str, path: Path
) -> None:
    """Ground-truth and learned test Gram matrices side by side (Fig. 5).

    Both matrices are min-max scaled to [0, 1] as in the authors' figure, and
    test points keep their dataset order (unsorted), as in the paper.
    """
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for axis, gram, title in zip(
        axes, (true_gram, learned_gram), (f"Ground truth {name}", f"Learned {name}")
    ):
        image = axis.imshow(_min_max(gram), vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("index")
        axis.set_ylabel("index")
    figure.colorbar(image, ax=axes, label="kernel value")
    figure.savefig(path)
    plt.close(figure)
