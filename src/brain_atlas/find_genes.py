import itertools
import logging
from collections import Counter, defaultdict
from typing import Dict, Sequence, Tuple, Union

import dask.array as da
import numba as nb
import numpy as np

from brain_atlas import Key
from brain_atlas.diff_exp import mannwhitneyu
from brain_atlas.util.tree import MultiNode

log = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, da.Array]


@nb.njit
def calc_log_fc(
    count_array: np.ndarray,
    sum_array: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
) -> np.ndarray:
    left_n = count_array[group1].sum()
    left_mean = sum_array[group1, :].sum(axis=0) / left_n

    right_n = count_array[group2].sum()
    right_mean = sum_array[group2, :].sum(axis=0) / right_n

    eps = 1 / (left_n + right_n)
    log_fc = np.log(left_mean + eps) - np.log(right_mean + eps)

    return log_fc


@nb.njit
def calc_nz(
    count_array: np.ndarray,
    nz_array: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
):
    left_n = count_array[group1].sum()
    left_nz = nz_array[group1, :].sum(axis=0) / left_n

    right_n = count_array[group2].sum()
    right_nz = nz_array[group2, :].sum(axis=0) / right_n

    return left_nz, right_nz


@nb.njit
def calc_filter(
    cluster_counts: np.ndarray,
    cluster_nz: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
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
    blocksize: int = 256000,
):
    """
    Calculates the number of nonzero elements per cluster.
    """
    n_cells, n_genes = data.shape

    cluster_nz_d = defaultdict(lambda: np.zeros(n_genes, dtype=np.uint32))

    log.debug("counting nonzero elements")
    for i in range(0, n_cells, blocksize):
        log.debug(f"{i} ...")
        cluster_i = clusters[i : i + blocksize]
        data = da.sign(data.counts[i : i + blocksize, :]).compute()
        for j in np.unique(cluster_i):
            cluster_nz_d[node_list[j]] += data[cluster_i == j, :].sum(0)

    cluster_nz_d = dict(cluster_nz_d)

    return cluster_nz_d


def de(
    data: ArrayLike,
    clusters: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
    gene_filter: np.ndarray,
    subsample: int = None,
):
    c_a = np.isin(clusters, group1)
    c_b = np.isin(clusters, group2)

    full_u = np.zeros(data.shape[1])
    full_p = np.zeros(data.shape[1])  # logp, no result = 0

    if np.any(gene_filter):
        ds_a = data[c_a, :][:, gene_filter]
        ds_b = data[c_b, :][:, gene_filter]
        if subsample is not None:
            ds_a = ds_a[calc_subsample(ds_a.shape[0], subsample), :]
            ds_b = ds_b[calc_subsample(ds_b.shape[0], subsample), :]

        if isinstance(data, da.Array):
            ds_a, ds_b = da.compute(ds_a, ds_b)

        u, logp = mannwhitneyu(ds_a, ds_b)

        full_u[gene_filter] = u
        full_p[gene_filter] = logp

    return full_u, full_p


def generic_de(
    get_comps,
    data: ArrayLike,
    clusters: np.ndarray,
    node_list: Sequence[Key],
    node_tree: Dict[Key, MultiNode],
    cluster_nz_d: Dict[Key, np.ndarray],
    de_results: Dict[Key, Tuple[np.ndarray, ...]],
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
    min_depth: int = 0,
):
    cluster_counts = Counter(clusters[clusters > -1])
    n_nodes = len(node_list)
    assert set(cluster_counts).issubset(range(n_nodes))
    assert all(node_list[i] in cluster_nz_d for i in cluster_counts)

    k2i = {k: i for i, k in enumerate(node_list)}

    def nd2i(nd):
        return k2i[nd.node_id]

    cluster_nz = np.zeros((n_nodes, data.shape[1]))
    for k in cluster_nz_d:
        cluster_nz[k2i[k], :] = cluster_nz_d[k]

    cluster_counts = np.array([cluster_counts[i] for i in range(n_nodes)])

    for k in node_list:
        if len(k) < min_depth:
            continue

        for comp, c_i, c_j in get_comps(k, node_tree, nd2i):
            if comp in de_results:
                continue

            log.debug(f"Comparing {c_i} with {c_j}")
            nz_diff, nz_filter = calc_filter(
                cluster_counts, cluster_nz, c_i, c_j, max_nz_b, delta_nz
            )

            _, p = de(data, clusters, c_i, c_j, nz_filter, subsample)
            de_results[comp] = p, nz_diff, nz_filter


def sibling_comps(k, node_tree, nd2i):
    if node_tree[k].is_leaf:
        return

    c_arrays = {
        nd.node_id: np.array(nd.pre_order(True, nd2i)) for nd in node_tree[k].children
    }

    for nd in node_tree[k].children:
        i = nd.node_id

        c_i = c_arrays[i]
        c_j = np.hstack([c_arrays[j] for j in c_arrays if i != j])

        yield i, c_i, c_j

        if len(node_tree[k].children) == 2:
            break


def pairwise_comps(k, node_tree, nd2i):
    if node_tree[k].is_leaf:
        return

    c_arrays = {
        nd.node_id: np.array(nd.pre_order(True, nd2i)) for nd in node_tree[k].children
    }

    for nd_i, nd_j in itertools.combinations(node_tree[k].children, 2):
        c_i = c_arrays[nd_i.node_id]
        c_j = c_arrays[nd_j.node_id]

        yield (nd_i.node_id, nd_j.node_id), c_i, c_j


def subtree_comps(k, node_tree, nd2i):
    if k == ():
        return

    below = np.array(node_tree[k].pre_order(True, nd2i))
    above = np.arange(len(node_tree))
    above = above[~np.isin(above, below)]

    yield k, below, above


def sibling_de(
    data: ArrayLike,
    clusters: np.ndarray,
    node_list: Sequence[Key],
    node_tree: Dict[Key, MultiNode],
    cluster_nz_d: Dict[Key, np.ndarray],
    sibling_results: Dict[Key, Tuple[np.ndarray, ...]] = None,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
    min_depth: int = 0,
):
    if sibling_results is None:
        sibling_results = dict()

    n_de = len(sibling_results)

    generic_de(
        sibling_comps,
        data,
        clusters,
        node_list,
        node_tree,
        cluster_nz_d,
        sibling_results,
        delta_nz,
        max_nz_b,
        subsample,
        min_depth,
    )

    for k in node_list:
        if len(k) < min_depth or node_tree[k].is_leaf:
            continue

        if len(node_tree[k].children) == 2:
            j = node_tree[k].children[1].node_id
            if j not in sibling_results:
                i = node_tree[k].children[0].node_id
                p, nz_diff, nz_filter = sibling_results[i]

                sibling_results[j] = p, -nz_diff, nz_filter

    return sibling_results, len(sibling_results) - n_de


def pairwise_sibling_de(
    data: ArrayLike,
    clusters: np.ndarray,
    node_list: Sequence[Key],
    node_tree: Dict[Key, MultiNode],
    cluster_nz_d: Dict[Key, np.ndarray],
    pairwise_results: Dict[Key, Tuple[np.ndarray, ...]] = None,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
    min_depth: int = 0,
):
    if pairwise_results is None:
        pairwise_results = dict()

    n_de = len(pairwise_results)

    generic_de(
        pairwise_comps,
        data,
        clusters,
        node_list,
        node_tree,
        cluster_nz_d,
        pairwise_results,
        delta_nz,
        max_nz_b,
        subsample,
        min_depth,
    )

    return pairwise_results, len(pairwise_results) - n_de


def subtree_de(
    data: ArrayLike,
    clusters: np.ndarray,
    node_list: Sequence[Key],
    node_tree: Dict[Key, MultiNode],
    cluster_nz_d: Dict[Key, np.ndarray],
    subtree_results: Dict[Key, Tuple[np.ndarray, ...]] = None,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
    min_depth: int = 0,
):
    if subtree_results is None:
        subtree_results = dict()

    n_de = len(subtree_results)

    generic_de(
        subtree_comps,
        data,
        clusters,
        node_list,
        node_tree,
        cluster_nz_d,
        subtree_results,
        delta_nz,
        max_nz_b,
        subsample,
        min_depth,
    )

    return subtree_results, len(subtree_results) - n_de


def get_de_totals(
    comp: Key,
    diff_results: Dict[Key, Tuple[np.ndarray, ...]],
    min_p: float = -10.0,
):
    if comp not in diff_results:
        return 0, 0

    p, nz_diff, nz_filter = diff_results[comp]

    total_a = ((p < min_p) & (nz_diff > 0)).sum()
    total_b = ((p < min_p) & (nz_diff < 0)).sum()

    return total_a, total_b


def collapse_tree(
    node_list: Sequence[Key],
    node_tree: Dict[Key, MultiNode],
    sib_results: Dict[Key, Tuple[np.ndarray, ...]],
    min_de: int = 5,
    min_p: float = -10.0,
):
    sib_totals = {k: get_de_totals(k, sib_results, min_p=min_p) for k in sib_results}

    exclude = set()
    new_leaf_list = []

    # assumes node_list is bottom-up. reverse it to start at root
    for k in node_list[::-1]:
        if k in exclude:
            continue

        if node_tree[k].is_leaf:
            # if we get to a leaf node, it qualifies
            new_leaf_list.append(k)
        elif not all(nd.node_id in sib_totals for nd in node_tree[k].children):
            # child nodes are not all tested, leave for now
            continue
        elif all(
            any(t < min_de for t in sib_totals[nd.node_id])
            for nd in node_tree[k].children
        ):
            # all child nodes fail DE test. but maybe grandchildren are interesting
            grandchildren = [
                all(t >= min_de for t in sib_totals[nd2.node_id])
                for nd in node_tree[k].children
                if not nd.is_leaf
                for nd2 in node_tree[nd.node_id].children
                if nd2.node_id in sib_totals
            ]

            if grandchildren and all(grandchildren):
                # this level was boring, but grandchildren passed
                continue
            else:
                # otherwise consider it a leaf and merge descendants
                new_leaf_list.append(k)
                exclude.update(node_tree[k].pre_order(True))

    return new_leaf_list
