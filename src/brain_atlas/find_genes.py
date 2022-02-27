import itertools
import logging
from collections import defaultdict
from typing import Any, Counter, Dict, Tuple, Union

import dask.array as da
import numba as nb
import numpy as np

from brain_atlas import Key
from brain_atlas.diff_exp import mannwhitneyu
from brain_atlas.util.tree import NodeTree

log = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, da.Array]
ResultsDict = Dict[Any, Tuple[np.ndarray, ...]]


@nb.njit
def calc_log_fc(
    count_array: np.ndarray,
    sum_array: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
) -> np.ndarray:
    n_1 = count_array[group1].sum()
    mu_1 = sum_array[group1, :].sum(axis=0) / n_1

    n_2 = count_array[group2].sum()
    mu_2 = sum_array[group2, :].sum(axis=0) / n_2

    eps = 1 / (n_1 + n_2)
    log_fc = np.log(mu_1 + eps) - np.log(mu_2 + eps)

    return log_fc


@nb.njit
def calc_nz(
    count_array: np.ndarray,
    nz_array: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
):
    n_1 = count_array[group1].sum()
    nz_1 = nz_array[group1, :].sum(axis=0) / n_1

    n_2 = count_array[group2].sum()
    nz_2 = nz_array[group2, :].sum(axis=0) / n_2

    return nz_1, nz_2


@nb.njit
def calc_filter(
    cluster_counts: np.ndarray,
    cluster_nz: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
    max_nz_b: float,
    delta_nz: float,
):
    nz_1, nz_2 = calc_nz(cluster_counts, cluster_nz, group1, group2)
    nz_filter = (np.minimum(nz_1, nz_2) < max_nz_b) & (np.abs(nz_1 - nz_2) > delta_nz)

    return nz_1, nz_2, nz_filter


def calc_subsample(n_samples: int, subsample: int):
    if n_samples <= subsample:
        return np.arange(n_samples)
    else:
        return np.sort(np.random.choice(n_samples, size=subsample, replace=False))


def cluster_nz_dict(
    data: da.Array, clusters: np.ndarray, node_tree: NodeTree, blocksize: int = 256000
):
    """
    Calculates the number of nonzero elements per cluster. The values in `clusters` must
    match the index attributes of the nodes in `node_tree`
    """
    n_cells, n_genes = data.shape

    ix_to_node = {node_tree[k].index: k for k in node_tree}
    cluster_nz_d = defaultdict(lambda: np.zeros(n_genes, dtype=np.uint32))

    log.debug("counting nonzero elements")
    for i in range(0, n_cells, blocksize):
        log.debug(f"{i} ...")
        cluster_i = clusters[i : i + blocksize]
        data = da.sign(data.counts[i : i + blocksize, :]).compute()
        for j in np.unique(cluster_i):
            cluster_nz_d[ix_to_node[j]] += data[cluster_i == j, :].sum(0)

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
    node_tree: NodeTree,
    cluster_nz_d: Dict[Key, np.ndarray],
    cluster_count_d: Counter[Key],
    de_results: ResultsDict,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
):
    n_nodes = len(node_tree)
    assert set(cluster_count_d).issubset(node_tree)
    assert set(cluster_count_d) == set(cluster_nz_d)

    cluster_nz = np.zeros((n_nodes, data.shape[1]))
    cluster_counts = np.zeros(n_nodes)
    for k in cluster_nz_d:
        cluster_nz[node_tree[k].index, :] = cluster_nz_d[k]
        cluster_counts[node_tree[k].index] = cluster_count_d[k]

    for k in node_tree:
        for comp, c_i, c_j in get_comps(k, node_tree):
            if comp in de_results:
                continue

            log.debug(f"Comparing {c_i} with {c_j}")
            nz_i, nz_j, nz_filter = calc_filter(
                cluster_counts, cluster_nz, c_i, c_j, max_nz_b, delta_nz
            )

            _, p = de(data, clusters, c_i, c_j, nz_filter, subsample)
            de_results[comp] = p, nz_i, nz_j, nz_filter


def sibling_comps(k: Key, node_tree: NodeTree):
    if node_tree[k].is_leaf:
        return

    c_arrays = {
        nd.node_id: np.array(nd.pre_order(True, lambda nd: nd.index))
        for nd in node_tree[k].children
    }

    for nd in node_tree[k].children:
        i = nd.node_id

        c_i = c_arrays[i]
        c_j = np.hstack([c_arrays[j] for j in c_arrays if i != j])

        yield i, c_i, c_j

        if len(node_tree[k].children) == 2:
            break


def pairwise_comps(k: Key, node_tree: NodeTree):
    if node_tree[k].is_leaf:
        return

    c_arrays = {
        nd.node_id: np.array(nd.pre_order(True, lambda nd: nd.index))
        for nd in node_tree[k].children
    }

    for nd_i, nd_j in itertools.combinations(node_tree[k].children, 2):
        c_i = c_arrays[nd_i.node_id]
        c_j = c_arrays[nd_j.node_id]

        yield (nd_i.node_id, nd_j.node_id), c_i, c_j


def subtree_comps(k: Key, node_tree: NodeTree):
    if k == ():
        return

    below = np.array(node_tree[k].pre_order(True, lambda nd: nd.index))
    above = np.arange(len(node_tree))
    above = above[~np.isin(above, below)]

    yield k, below, above


def sibling_de(
    data: ArrayLike,
    clusters: np.ndarray,
    node_tree: NodeTree,
    cluster_nz_d: Dict[Key, np.ndarray],
    cluster_count_d: Counter[Key],
    sibling_results: ResultsDict = None,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
):
    if sibling_results is None:
        sibling_results = dict()

    n_de = len(sibling_results)

    generic_de(
        sibling_comps,
        data,
        clusters,
        node_tree,
        cluster_nz_d,
        cluster_count_d,
        sibling_results,
        delta_nz,
        max_nz_b,
        subsample,
    )

    for k in node_tree:
        if node_tree[k].is_leaf:
            continue

        if len(node_tree[k].children) == 2:
            j = node_tree[k].children[1].node_id
            if j not in sibling_results:
                i = node_tree[k].children[0].node_id
                p, nz_i, nz_j, nz_filter = sibling_results[i]

                sibling_results[j] = p, nz_j, nz_i, nz_filter

    return sibling_results, len(sibling_results) - n_de


def pairwise_sibling_de(
    data: ArrayLike,
    clusters: np.ndarray,
    node_tree: NodeTree,
    cluster_nz_d: Dict[Key, np.ndarray],
    cluster_count_d: Counter[Key],
    pairwise_results: ResultsDict = None,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
):
    if pairwise_results is None:
        pairwise_results = dict()

    n_de = len(pairwise_results)

    generic_de(
        pairwise_comps,
        data,
        clusters,
        node_tree,
        cluster_nz_d,
        cluster_count_d,
        pairwise_results,
        delta_nz,
        max_nz_b,
        subsample,
    )

    return pairwise_results, len(pairwise_results) - n_de


def subtree_de(
    data: ArrayLike,
    clusters: np.ndarray,
    node_tree: NodeTree,
    cluster_nz_d: Dict[Key, np.ndarray],
    cluster_count_d: Counter[Key],
    subtree_results: ResultsDict = None,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
):
    if subtree_results is None:
        subtree_results = dict()

    n_de = len(subtree_results)

    generic_de(
        subtree_comps,
        data,
        clusters,
        node_tree,
        cluster_nz_d,
        cluster_count_d,
        subtree_results,
        delta_nz,
        max_nz_b,
        subsample,
    )

    return subtree_results, len(subtree_results) - n_de


def get_de_totals(comp: Key, diff_results: ResultsDict, min_p: float = -10.0):
    if comp not in diff_results:
        return 0, 0

    p, nz_i, nz_j, nz_filter = diff_results[comp]
    nz_diff = nz_i - nz_j

    total_a = ((p < min_p) & (nz_diff > 0)).sum()
    total_b = ((p < min_p) & (nz_diff < 0)).sum()

    return total_a, total_b


def collapse_tree(
    node_tree: NodeTree,
    sib_results: ResultsDict,
    min_de: int = 5,
    min_p: float = -10.0,
):
    sib_totals = {k: get_de_totals(k, sib_results, min_p=min_p) for k in sib_results}

    exclude = set()
    new_leaf_list = []

    # sort by length of key to get top-down ordering
    for k in sorted(node_tree, key=len):
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
