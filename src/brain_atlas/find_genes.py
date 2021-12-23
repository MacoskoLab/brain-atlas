import logging
from collections import Counter
from typing import Dict, Sequence, Tuple, Union

import dask.array as da
import numpy as np

import brain_atlas.util.tree as tree
from brain_atlas import Key
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


def calc_subsample(n_samples: int, subsample: int):
    if n_samples <= subsample:
        return np.arange(n_samples)
    else:
        return np.sort(np.random.choice(n_samples, size=subsample, replace=False))


def cluster_nz_dict(
    data: da.Array,
    clusters: np.ndarray,
    node_list: Sequence[Key],
    cluster_nz_d: Dict[Key, np.ndarray] = None,
):
    """
    Calculates the number of nonzero elements per cluster. If given, an existing
    dictionary of arrays can be updated, under the assumption that the rest of the tree
    is unchanged.
    """
    if cluster_nz_d is None:
        cluster_nz_d = dict()

    k2i = {k: i for i, k in enumerate(node_list)}

    # first need to remove parents to update their nz array as well
    for k in node_list:
        if len(k) and k not in cluster_nz_d and k[:-1] in cluster_nz_d:
            cluster_nz_d.pop(k[:-1])

    for k in node_list:
        if k not in cluster_nz_d:
            cluster_i = clusters == k2i[k]
            if cluster_i.any():
                cluster_nz_d[k] = da.sign(data[cluster_i, :]).sum(0).compute()

    return cluster_nz_d


def de(
    ds: Union[np.ndarray, da.Array],
    clusters: np.ndarray,
    group1: Sequence[int],
    group2: Sequence[int],
    gene_filter: np.ndarray,
    subsample: int = None,
):
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    c_a = np.isin(clusters, group1)
    c_b = np.isin(clusters, group2)

    full_u = np.zeros(ds.shape[1])
    full_p = np.zeros(ds.shape[1])  # logp, no result = 0

    if np.any(gene_filter):
        ds_a = ds[c_a, :]
        ds_b = ds[c_b, :]
        if subsample is not None:
            ds_a = ds_a[calc_subsample(ds_a.shape[0], subsample), :]
            ds_b = ds_b[calc_subsample(ds_b.shape[0], subsample), :]

        if isinstance(ds, da.Array):
            ds_a, ds_b = da.compute(ds_a, ds_b)

        u, logp = mannwhitneyu(ds_a[:, gene_filter], ds_b[:, gene_filter])

        full_u[gene_filter] = u
        full_p[gene_filter] = logp

    return full_u, full_p


def leiden_tree_diff_exp(
    data: da.array,
    clusters: np.ndarray,
    node_list: Sequence[Key],
    node_tree: Dict[Key, tree.MultiNode],
    cluster_nz_d: Dict[Key, np.ndarray] = None,
    sibling_results: Dict[Key, Tuple[np.ndarray, ...]] = None,
    subtree_results: Dict[Key, Tuple[np.ndarray, ...]] = None,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
):
    cluster_counts = Counter(clusters[clusters > -1])
    n_nodes = len(node_list)
    ni_set = set(range(n_nodes))
    assert set(cluster_counts).issubset(ni_set)

    k2i = {k: i for i, k in enumerate(node_list)}

    def nd2i(nd):
        return k2i[nd.node_id]

    if cluster_nz_d is None:
        cluster_nz_d = cluster_nz_dict(data, clusters, node_list)

    assert all(node_list[i] in cluster_nz_d for i in cluster_counts)

    cluster_nz = np.zeros((n_nodes, data.shape[1]))
    for k in cluster_nz_d:
        cluster_nz[k2i[k], :] = cluster_nz_d[k]

    cluster_counts = np.array([cluster_counts[i] for i in range(n_nodes)])

    if sibling_results is None:
        sibling_results = dict()
    if subtree_results is None:
        subtree_results = dict()

    n_sib = len(sibling_results)
    n_sub = len(subtree_results)

    for k in node_list:
        if not node_tree[k].is_leaf:
            c_lists = {
                nd.node_id: nd.pre_order(True, nd2i) for nd in node_tree[k].children
            }

            for n in node_tree[k].children:
                i = n.node_id
                if i in sibling_results:
                    continue

                c_i = c_lists[i]
                c_j = [c_o for j in c_lists if i != j for c_o in c_lists[j]]

                # don't calculate if it's redundant with a 1-vs-all comp
                if k == () and (len(c_i) == 1 or len(c_j) == 1):
                    continue

                log.debug(f"Comparing {c_i} with {c_j}")
                nz_diff, nz_filter = calc_filter(
                    cluster_counts, cluster_nz, c_i, c_j, max_nz_b, delta_nz
                )

                _, p = de(data, clusters, c_i, c_j, nz_filter, subsample)
                sibling_results[i] = p, nz_diff, nz_filter

                # common special case is 2 siblings: results are symmetrical
                if len(node_tree[k].children) == 2:
                    j = node_tree[k].children[1].node_id
                    sibling_results[j] = p, -nz_diff, nz_filter
                    break

        if k != ():
            if k in subtree_results:
                continue

            below = node_tree[k].pre_order(True, nd2i)
            above = sorted(ni_set - set(below))

            # don't calculate redundant comparison
            if len(above) == 1:
                continue

            log.debug(f"Comparing {below} with {above}")
            nz_diff, nz_filter = calc_filter(
                cluster_counts, cluster_nz, below, above, max_nz_b, delta_nz
            )

            _, p = de(data, clusters, below, above, nz_filter, subsample)
            subtree_results[k] = p, nz_diff, nz_filter

    return (
        sibling_results,
        subtree_results,
        len(sibling_results) - n_sib,
        len(subtree_results) - n_sub,
    )
