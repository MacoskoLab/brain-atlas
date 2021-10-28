import logging
from collections import Counter
from pathlib import Path
from typing import Sequence

import click
import dask
import dask.array as da
import numpy as np
import scipy.cluster.hierarchy

from brain_atlas.diff_exp import mannwhitneyu
from brain_atlas.leiden_tree import LeidenTree
from brain_atlas.util.dataset import Dataset

# from numcodecs import Blosc


log = logging.getLogger(__name__)


def calc_log_fc(
    count_array: np.ndarray,
    sum_array: np.ndarray,
    group1: Sequence[int],
    group2: Sequence[int],
) -> np.ndarray:
    left_n = count_array[group1].sum()
    left_mean = sum_array[group1, :].sum(axis=0) / left_n

    right_n = count_array[group2].sum()
    right_mean = sum_array[group2, :].sum(axis=0) / right_n

    eps = 1 / (left_n + right_n)
    log_fc = np.abs(np.log(left_mean + eps) - np.log(right_mean + eps))

    return log_fc


def de(
    ds: da.array,
    clusters: np.ndarray,
    group1: Sequence[int],
    group2: Sequence[int],
    gene_filter: np.ndarray,
):
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    c_a = np.isin(clusters, group1)
    c_b = np.isin(clusters, group2)

    ds_a = ds[c_a, :].compute()
    ds_b = ds[c_b, :].compute()

    full_u = np.zeros(ds.shape[1])
    full_p = np.zeros(ds.shape[1])  # logp, no result = 0

    u, logp = mannwhitneyu(ds_a[:, gene_filter], ds_b[:, gene_filter])

    full_u[gene_filter] = u
    full_p[gene_filter] = logp

    return full_u, full_p


def hierarchical_diff_exp(data, clusters: np.ndarray, min_fc: float = 1.25):
    n_cells = data.shape[0]
    cluster_counts = Counter(clusters)
    min_log_fc = np.log(min_fc)

    cluster_sums = {}
    for i in cluster_counts:
        if cluster_counts[i] > np.sqrt(n_cells):
            cluster_sums[i] = data[clusters == i, :].sum(0).compute()

    n_clusters = len(cluster_sums)
    cluster_sum_array = np.vstack([cluster_sums[i] for i in range(n_clusters)])
    cluster_count_array = np.array([cluster_counts[i] for i in range(n_clusters)])

    z = scipy.cluster.hierarchy.linkage(
        cluster_sum_array / cluster_count_array[:, None],
        method="average",
        metric="cosine",
    )
    root, rd = scipy.cluster.hierarchy.to_tree(z, rd=True)

    diff_results = {}

    for i in range(0, 2 * n_clusters - 1):
        if i >= n_clusters:
            left_child = rd[i].get_left()
            left = left_child.pre_order(lambda x: x.id)

            right_child = rd[i].get_right()
            right = right_child.pre_order(lambda x: x.id)

            # don't calculate if it's redundant with a 1-vs-all comp
            if i == 2 * n_clusters - 2 and (len(left) == 1 or len(right) == 1):
                continue

            log.debug(f"Comparing {left} with {right}")

            log_fc = calc_log_fc(cluster_count_array, cluster_sum_array, left, right)

            u, p = de(data, clusters, left, right, log_fc > min_log_fc)
            diff_results[tuple(left), tuple(right)] = u, p, log_fc

        if i < 2 * n_clusters - 2:
            below = rd[i].pre_order(lambda x: x.id)
            above = [j for j in range(n_clusters) if j not in below]

            # don't calculate redundant comparison
            if len(above) == 1:
                continue

            log.debug(f"Comparing {below} with {above}")

            log_fc = calc_log_fc(cluster_count_array, cluster_sum_array, below, above)

            u, p = de(data, clusters, below, above, log_fc > min_log_fc)
            diff_results[tuple(below), tuple(above)] = u, p, log_fc

    return diff_results


@click.command("find_genes")
@click.argument("root_path", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("level", type=int, nargs=-1)
@click.option(
    "-r", "--recursive", is_flag=True, help="Descend into subcluster directories"
)
@click.option(
    "--min-fc", type=float, default=1.25, help="Minimum fold-change for testing"
)
@click.option("--overwrite", is_flag=True, help="Don't use any cached results")
def main(
    root_path: str,
    level: Sequence[int],
    min_fc: float = 1.25,
):
    """
    Loads selected genes from LEVEL of the ROOT_PATH Leiden tree and performs
    Mann-Whitney U tests to identify significant markers for each cluster
    """

    root = LeidenTree.from_path(Path(root_path))
    ds = Dataset(str(root.data))

    tree = LeidenTree.from_path(root.subcluster_path(level))

    clusters = np.load(tree.clustering)
    if tree.resolution is None:
        resolution = max(clusters, key=float)
    else:
        resolution = tree.resolution

    log.debug(f"Using clustering with resolution {resolution}")
    clusters = clusters[resolution]
    assert clusters.shape[0] == ds.counts.shape[0], "Clusters do not match input data"

    ci = clusters > -1
    n_cells = ci.sum()
    ci_clusters = clusters[ci]

    log.info(f"Processing {n_cells} / {clusters.shape[0]} cells from {tree.data}")
    d_i = ds.counts[ci, :]

    log.debug(f"Loading selected genes from {tree.selected_genes}")
    selected_genes = np.load(tree.selected_genes)["selected_genes"]
    n_genes = selected_genes.shape[0]

    # subselect genes
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        d_i_mem = d_i[:, selected_genes].rechunk({1: n_genes}).persist()

    hierarchical_diff_exp(d_i_mem, ci_clusters, min_fc)
