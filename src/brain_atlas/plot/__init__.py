from contextlib import contextmanager

import matplotlib.colors
import matplotlib.figure
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


@contextmanager
def new_fig(pdf_pages: PdfPages, figsize=(11, 8.5)):
    fig = matplotlib.figure.Figure(figsize=figsize)

    yield fig

    pdf_pages.savefig(fig)


@contextmanager
def new_ax(
    pdf_pages: PdfPages,
    include_fig=False,
    ax_bounds=(0.05, 0.05, 0.9, 0.9),
    figsize=(11, 8.5),
):
    with new_fig(pdf_pages, figsize) as fig:
        ax: Axes = fig.add_axes(ax_bounds)

        if include_fig:
            yield fig, ax
        else:
            yield ax


def plot_gene_exp(exp_pct_nz, pct, pdf_pages):
    with new_fig(pdf_pages, figsize=(16, 8)) as fig:
        gs = GridSpec(1, 2, wspace=0.05)

        ax0 = fig.add_subplot(gs[0])
        ax0.scatter(
            exp_pct_nz,
            pct,
            alpha=0.8,
            s=1,
            c=(pct < exp_pct_nz - 0.05),
            cmap="viridis_r",
        )
        ax0.axis("equal")
        ax0.set_xlabel("Expected cell proportion")
        ax0.set_ylabel("Observed cell proportion")

        ax1 = fig.add_subplot(gs[1])
        ax1.hist(
            np.clip(exp_pct_nz - pct, 0, 1), bins=np.linspace(0, 1.0, 101), log=True
        )


def plot_marker_gene(x, marker_gene_dist, ix, title, pdf_pages, pct=95):
    with new_ax(pdf_pages, include_fig=True) as (fig, ax):
        c = ax.scatter(
            x[ix, 0],
            x[ix, 1],
            s=0.1,
            alpha=0.3,
            c=marker_gene_dist[ix],
            rasterized=True,
            cmap="Blues",
            norm=matplotlib.colors.Normalize(
                0, max(np.percentile(marker_gene_dist, pct), 1), clip=True
            ),
        )
        ax.set_title(title)
        ax.tick_params(labelbottom=False, labelleft=False)
        fig.colorbar(c, ax=ax)


def plot_cluster(x, labels, label_to_plot, ix, title, pdf_pages):
    with new_ax(pdf_pages) as ax:
        b = np.array([int(labels[i] == label_to_plot) for i in ix])
        ax.scatter(
            x[ix, 0],
            x[ix, 1],
            s=0.1 + 2.4 * b,
            alpha=0.8,
            c=b,
            cmap="Blues",
            rasterized=True,
        )
        ax.set_title(title)
        ax.tick_params(labelbottom=False, labelleft=False)


def plot_spotmap(clusters, max_c, marker_genes, pdf_pages):
    with new_ax(pdf_pages) as ax:
        x, y = zip(*np.ndindex(len(marker_genes), max_c))

        s = []
        for g in sorted(marker_genes):
            for i in range(max_c):
                s.append(
                    512 * marker_genes[g][clusters == i].mean() / marker_genes[g].mean()
                )

        ax.scatter(x, y, s=np.sqrt(s))

        ax.set_xticks(range(len(marker_genes)))
        ax.set_xticklabels(sorted(marker_genes), rotation="vertical")

        ax.tick_params(top=True, labeltop=True)

        ax.set_yticks(range(max_c))

        ax.set_xlim(-0.5, len(marker_genes) - 0.5)
        ax.set_ylim(-0.5, max_c - 0.5)
