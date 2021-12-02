import logging
from collections import Counter
from typing import Sequence

import dask
import dask.array as da
import numpy as np
import scipy.cluster.hierarchy

from brain_atlas.diff_exp import mannwhitneyu
from brain_atlas.leiden_tree import LeidenTree
from brain_atlas.util.dataset import Dataset

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


def calc_nz(
    count_array: np.ndarray,
    nz_array: np.ndarray,
    group1: Sequence[int],
    group2: Sequence[int],
):
    left_n = count_array[group1].sum()
    left_nz = nz_array[group1, :].sum(axis=0) / left_n

    right_n = count_array[group2].sum()
    right_nz = nz_array[group2, :].sum(axis=0) / right_n

    return left_nz, right_nz


def calc_filter(
    cluster_counts: np.ndarray,
    cluster_nz: np.ndarray,
    group1: Sequence[int],
    group2: Sequence[int],
    max_nz_b: float,
    delta_nz: float,
):
    nzA, nzB = calc_nz(cluster_counts, cluster_nz, group1, group2)
    min_nz = np.minimum(nzA, nzB)
    nz_diff = nzA - nzB
    nz_filter = (min_nz < max_nz_b) & (np.abs(nz_diff) > delta_nz)

    return nz_diff, nz_filter


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

    full_u = np.zeros(ds.shape[1])
    full_p = np.zeros(ds.shape[1])  # logp, no result = 0

    if np.any(gene_filter):
        ds_a = ds[c_a, :].compute()
        ds_b = ds[c_b, :].compute()

        u, logp = mannwhitneyu(ds_a[:, gene_filter], ds_b[:, gene_filter])

        full_u[gene_filter] = u
        full_p[gene_filter] = logp

    return full_u, full_p


def hierarchical_diff_exp(
    data, clusters: np.ndarray, delta_nz: float = 0.2, max_nz_b: float = 0.2
):
    cluster_counts = Counter(clusters[clusters > -1])
    n_clusters = len(cluster_counts)
    assert sorted(range(n_clusters)) == sorted(cluster_counts)

    cluster_sums = {}
    cluster_nz = {}

    for i in cluster_counts:
        cluster_sums[i] = data[clusters == i, :].sum(0).compute()
        cluster_nz[i] = da.sign(data[clusters == i, :]).sum(0).compute()

    cluster_sums = np.vstack([cluster_sums[i] for i in range(n_clusters)])
    cluster_nz = np.vstack([cluster_nz[i] for i in range(n_clusters)])
    cluster_counts = np.array([cluster_counts[i] for i in range(n_clusters)])

    z = scipy.cluster.hierarchy.linkage(
        cluster_sums / cluster_counts[:, None],
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
            nz_diff, nz_filter = calc_filter(
                cluster_counts, cluster_nz, left, right, max_nz_b, delta_nz
            )

            u, p = de(data, clusters, left, right, nz_filter)
            diff_results[tuple(left), tuple(right)] = u, p, nz_diff, nz_filter

        if i < 2 * n_clusters - 2:
            below = rd[i].pre_order(lambda x: x.id)
            above = [j for j in range(n_clusters) if j not in below]

            # don't calculate redundant comparison
            if len(above) == 1:
                continue

            log.debug(f"Comparing {below} with {above}")
            nz_diff, nz_filter = calc_filter(
                cluster_counts, cluster_nz, below, above, max_nz_b, delta_nz
            )

            u, p = de(data, clusters, below, above, nz_filter)
            diff_results[tuple(below), tuple(above)] = u, p, nz_diff, nz_filter

    return z, diff_results


def process_tree(
    tree: LeidenTree,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    selected_only: bool = True,
):
    ds = Dataset(tree.data)

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

    if selected_only:
        log.debug(f"Loading selected genes from {tree.selected_genes}")
        selected_genes = np.load(tree.selected_genes)["selected_genes"]
        n_genes = selected_genes.shape[0]

        # subselect genes
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            d_i_mem = d_i[:, selected_genes].rechunk({1: n_genes}).persist()
    else:
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            d_i_mem = d_i.persist()

    return hierarchical_diff_exp(d_i_mem, ci_clusters, delta_nz, max_nz_b)
