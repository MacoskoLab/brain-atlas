import logging
from pathlib import Path
from typing import Sequence

import click

# import dask
# import dask.array as da
import numpy as np

# from brain_atlas.diff_exp import mannwhitneyu
from brain_atlas.leiden_tree import LeidenTree
from brain_atlas.util.dataset import Dataset

# from numcodecs import Blosc


log = logging.getLogger(__name__)


@click.command("find_genes")
@click.argument("root_path", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("level", type=int, nargs=-1)
@click.option(
    "-r", "--recursive", is_flag=True, help="Descend into subcluster directories"
)
@click.option(
    "--min-fc", type=float, default=0.0, help="Minimum fold-change (log2) for testing"
)
@click.option("--overwrite", is_flag=True, help="Don't use any cached results")
def main(
    root_path: str,
    level: Sequence[int],
    # min_fc: float = 0.0,
    # overwrite: bool = False,
):
    """
    Loads selected genes from LEVEL of the ROOT_PATH Leiden tree and performs
    Mann-Whitney U tests to identify significant markers for each cluster
    """

    root = LeidenTree.from_path(Path(root_path))
    ds = Dataset(str(root.data))

    tree = LeidenTree.from_path(root.subcluster_path(level))
    # open the tree one level up (this might be the root)
    parent = LeidenTree.from_path(root.subcluster_path(level[:-1]))

    log.debug(f"Using parent clustering with resolution {tree.resolution}")
    clusters = np.load(parent.clustering)[tree.resolution]
    assert clusters.shape[0] == ds.counts.shape[0], "Clusters do not match input data"
