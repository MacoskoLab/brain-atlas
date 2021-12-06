import logging
from collections import Counter
from typing import Dict, Sequence

import dask.array as da
import numpy as np
import scipy.cluster.hierarchy

import brain_atlas.util.tree as tree
from brain_atlas.diff_exp import mannwhitneyu

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


def leiden_tree_diff_exp(
    data: da.array,
    clusters: np.ndarray,
    rd: Dict[int, tree.ClusterNode],
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
):
    cluster_counts = Counter(clusters[clusters > -1])
    n_nodes = len(rd)
    assert set(cluster_counts).issubset(rd)

    cluster_nz = np.zeros((n_nodes, data.shape[1]))
    for i in range(n_nodes):
        cluster_i = clusters == i
        if cluster_i.any():
            cluster_nz[i, :] = da.sign(data[cluster_i, :]).sum(0).compute()

    cluster_counts = np.array([cluster_counts[i] for i in range(n_nodes)])
    rd_set = set(rd)

    sibling_results = {}
    subtree_results = {}

    for i in range(n_nodes):
        if not rd[i].is_leaf:
            c_lists = {n.node_id: n.pre_order(True) for n in rd[i].children}

            for n in rd[i].children:
                j = n.node_id
                c_j = c_lists[j]
                c_other = [c_o for k in c_lists if j != k for c_o in c_lists[k]]

                # don't calculate if it's redundant with a 1-vs-all comp
                if i == n_nodes - 1 and (len(c_j) == 1 or len(c_other) == 1):
                    continue

                # log.debug(f"Comparing {c_this} with {c_other}")
                nz_diff, nz_filter = calc_filter(
                    cluster_counts, cluster_nz, c_j, c_other, max_nz_b, delta_nz
                )

                _, p = de(data, clusters, c_j, c_other, nz_filter)
                sibling_results[j] = p, nz_diff, nz_filter

                # special case for 2 siblings: results are symmetrical
                if len(rd[i].children) == 2:
                    k = rd[i].children[1].node_id
                    sibling_results[k] = p, -nz_diff, nz_filter
                    break

        if i != n_nodes - 1:
            below = rd[i].pre_order(True)
            below_set = set(below)
            above = sorted(rd_set - below_set)

            # don't calculate redundant comparison
            if len(above) == 1:
                continue

            # log.debug(f"Comparing {below} with {above}")
            nz_diff, nz_filter = calc_filter(
                cluster_counts, cluster_nz, below, above, max_nz_b, delta_nz
            )

            _, p = de(data, clusters, below, above, nz_filter)
            subtree_results[i] = p, nz_diff, nz_filter

    return sibling_results, subtree_results


def hierarchical_diff_exp(
    data, clusters: np.ndarray, delta_nz: float = 0.2, max_nz_b: float = 0.2
):
    cluster_counts = Counter(clusters[clusters > -1])
    n_clusters = len(cluster_counts)
    assert list(range(n_clusters)) == sorted(cluster_counts)

    cluster_sums = {}
    cluster_nz = {}

    for i in cluster_counts:
        cluster_sums[i] = data[clusters == i, :].sum(0).compute()
        cluster_nz[i] = da.sign(data[clusters == i, :]).sum(0).compute()

    cluster_sums = np.vstack([cluster_sums[i] for i in range(n_clusters)])
    cluster_nz = np.vstack([cluster_nz[i] for i in range(n_clusters)])
    cluster_counts = np.array([cluster_counts[i] for i in range(n_clusters)])

    z = scipy.cluster.hierarchy.linkage(
        cluster_sums / cluster_counts[:, None], method="average", metric="cosine"
    )
    root, rd = scipy.cluster.hierarchy.to_tree(z, rd=True)

    sib_results = {}
    sub_results = {}

    for i in range(0, 2 * n_clusters - 1):
        if i >= n_clusters:
            left_child = rd[i].get_left()
            left = left_child.pre_order()

            right_child = rd[i].get_right()
            right = right_child.pre_order()

            # don't calculate if it's redundant with a 1-vs-all comp
            if i == 2 * n_clusters - 2 and (len(left) == 1 or len(right) == 1):
                continue

            log.debug(f"Comparing {left} with {right}")
            nz_diff, nz_filter = calc_filter(
                cluster_counts, cluster_nz, left, right, max_nz_b, delta_nz
            )

            _, p = de(data, clusters, left, right, nz_filter)
            sib_results[i] = p, nz_diff, nz_filter

        if i < 2 * n_clusters - 2:
            below = rd[i].pre_order()
            above = [j for j in range(n_clusters) if j not in below]

            # don't calculate redundant comparison
            if len(above) == 1:
                continue

            log.debug(f"Comparing {below} with {above}")
            nz_diff, nz_filter = calc_filter(
                cluster_counts, cluster_nz, below, above, max_nz_b, delta_nz
            )

            _, p = de(data, clusters, below, above, nz_filter)
            sub_results[i] = p, nz_diff, nz_filter

    return z, sib_results, sub_results
