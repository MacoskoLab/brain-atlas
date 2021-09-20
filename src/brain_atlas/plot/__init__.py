import matplotlib.colors
import matplotlib.figure
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec


def plot_gene_exp(exp_pct_nz, pct, output_file):
    fig = matplotlib.figure.Figure(figsize=(16, 8))
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 2, wspace=0.05)

    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(
        exp_pct_nz, pct, alpha=0.8, s=1, c=(pct < exp_pct_nz - 0.05), cmap="viridis_r"
    )
    ax0.axis("equal")
    ax0.set_xlabel("Expected cell proportion")
    ax0.set_ylabel("Observed cell proportion")

    ax1 = fig.add_subplot(gs[1])
    ax1.hist(np.clip(exp_pct_nz - pct, 0, 1), bins=np.linspace(0, 1.0, 101), log=True)

    FigureCanvasAgg(fig).print_figure(output_file)
