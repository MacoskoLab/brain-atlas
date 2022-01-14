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
    include_index: bool = False,
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
    max_yc = int(np.ceil(np.log10(max(y_counts[i] for i in ix)))) + 1
    max_yc += max_yc // 3

    ax.set_yticks(np.arange(ix.shape[0]))
    ax.set_yticklabels(
        [f"{y_labels[i]:<{max_y}} {y_counts[i]:>{max_yc},}" for i in ix],
        fontsize="small",
        fontdict={"fontfamily": "monospace"},
    )

    max_x = max(map(len, x_labels.values()))
    max_xc = int(np.ceil(np.log10(max(x_counts[i] for i in ix2)))) + 1
    max_xc += max_xc // 3

    if include_index:
        xtick_labels = [
            f"{x_labels[i]:<{max_x}} {x_counts[i]:>{max_xc},} {i:>4}" for i in ix2
        ]
    else:
        xtick_labels = [f"{x_labels[i]:<{max_x}} {x_counts[i]:>{max_xc},}" for i in ix2]

    ax.set_xticks(np.arange(ix2.shape[0]))
    ax.set_xticklabels(
        xtick_labels,
        fontsize="small",
        fontdict={"fontfamily": "monospace"},
        rotation=90,
    )

    ax.tick_params(axis="x", labeltop=True, labelbottom=True)
    ax.grid(True)

    return fig
