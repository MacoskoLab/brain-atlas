from typing import Dict

import matplotlib.colors
import matplotlib.figure
import numpy as np


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    ix: np.ndarray,
    ix2: np.ndarray,
    x_labels: Dict[int, str],
    x_counts: Dict[int, int],
    y_labels: Dict[int, str],
    y_counts: Dict[int, int],
):
    fig = matplotlib.figure.Figure(figsize=(32, 32))
    ax = fig.add_axes((0.05, 0.05, 0.9, 0.9))

    ax.matshow(
        confusion_matrix[np.ix_(ix, ix2)],
        cmap="Blues",
        norm=matplotlib.colors.Normalize(0, 1),
        interpolation="none",
    )

    max_y = max(map(len, y_labels.values()))
    ax.set_yticks(np.arange(ix.shape[0]))
    ax.set_yticklabels(
        [f"{y_labels[i]:<{max_y}} {y_counts[i]:>6}" for i in ix],
        fontsize="small",
        fontdict={"fontfamily": "monospace"},
    )

    max_x = max(map(len, x_labels.values()))
    ax.set_xticks(np.arange(ix2.shape[0]))
    ax.set_xticklabels(
        [f"{x_labels[i]:<{max_x}} {x_counts[i]:>6} {i:>4}" for i in ix2],
        fontsize="small",
        fontdict={"fontfamily": "monospace"},
        rotation=90,
    )

    ax.tick_params(axis="x", labeltop=True, labelbottom=True)
    ax.grid(True)

    return fig
