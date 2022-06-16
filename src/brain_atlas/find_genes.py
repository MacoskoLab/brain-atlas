import itertools
import logging
from typing import Dict, Hashable, Tuple, Union

import dask.array as da
import numba as nb
import numpy as np

from brain_atlas import Key
from brain_atlas.diff_exp import mannwhitneyu
from brain_atlas.util.tree import NodeTree

log = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, da.Array]
DiffExpResult = Tuple[np.ndarray, ...]
ResultsDict = Dict[Hashable, DiffExpResult]


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


def cluster_reduce(
    compute_func,
    data: da.Array,
    n_nodes: int,
    clusters: np.ndarray,
    blocksize: int = 256000,
) -> np.ndarray:
    """
    Generic function to perform some operation on a count array and then sum up
    the result per-cluster.
    """

    n_cells, n_genes = data.shape

    cluster_arr = np.zeros((n_nodes, n_genes), dtype=np.uint32)

    log.debug("counting elements")
    for i in range(0, n_cells, blocksize):
        log.debug(f"{i} ...")
        cluster_i = clusters[i : i + blocksize]
        data_i = compute_func(data[i : i + blocksize, :])
        for j in np.unique(cluster_i):
            cluster_arr[j] += data_i[cluster_i == j, :].sum(0)

    return cluster_arr


def cluster_nz_arr(
    data: da.Array, n_nodes: int, clusters: np.ndarray, blocksize: int = 256000
) -> np.ndarray:
    """
    Calculates the number of nonzero elements per cluster. Returns an array with
    shape (n_nodes, n_genes) where row i corresponds to cluster i
    """

    def nz_func(d: da.Array):
        return da.sign(d).compute()

    return cluster_reduce(nz_func, data, n_nodes, clusters, blocksize)


def cluster_sum_arr(
    data: da.Array, n_nodes: int, clusters: np.ndarray, blocksize: int = 256000
) -> np.ndarray:
    """
    Calculates the total gene counts per cluster. Returns an array with
    shape (n_nodes, n_genes) where row i corresponds to cluster i
    """

    def sum_func(d: da.Array):
        return da.compute(d)[0]

    return cluster_reduce(sum_func, data, n_nodes, clusters, blocksize)


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
    cluster_nz: np.ndarray,
    cluster_counts: np.ndarray,
    de_results: ResultsDict,
    delta_nz: float = 0.2,
    max_nz_b: float = 0.2,
    subsample: int = None,
):
    assert cluster_nz.shape[0] == cluster_counts.shape[0]
    assert np.array_equal(cluster_nz.sum(1).nonzero()[0], cluster_counts.nonzero()[0])

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
    cluster_nz: np.ndarray,
    cluster_counts: np.ndarray,
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
        cluster_nz,
        cluster_counts,
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
    cluster_nz: np.ndarray,
    cluster_counts: np.ndarray,
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
        cluster_nz,
        cluster_counts,
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
    cluster_nz: np.ndarray,
    cluster_counts: np.ndarray,
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
        cluster_nz,
        cluster_counts,
        subtree_results,
        delta_nz,
        max_nz_b,
        subsample,
    )

    return subtree_results, len(subtree_results) - n_de


def filter_res(de_res: DiffExpResult, max_p: float, max_nz_b: float):
    """
    Refilter a DE result with a maximum p-value and max nz% for the lower side
    """
    p, nz_1, nz_2, nzf = de_res
    nzd = nz_1 - nz_2

    min_nz = np.minimum(nz_1, nz_2)
    nzf_2 = min_nz < max_nz_b
    assert not np.any(nzf < nzf_2), "New nz% filter has a larger gene selection"

    nzf = nzf & (p < max_p) & nzf_2

    return nzf, nzd


def get_de_count(de_res: DiffExpResult, max_p: float = -10.0, max_nz_b: float = 0.2):
    """
    Retrieves the number of DE genes (in each direction) for a given DE result
    """
    nzf, nzd = filter_res(de_res, max_p, max_nz_b)

    return (nzd[nzf] > 0).sum(), (nzd[nzf] < 0).sum()


def get_gene_lists(
    de_res: DiffExpResult,
    gene_list: list,
    top_n: int = 20,
    max_p: float = -10.0,
    max_nz_b: float = 0.2,
):
    """
    Retrieves the top genes for a given DE result by indexing
    into a list of names or ids
    """

    nzf, nzd = filter_res(de_res, max_p, max_nz_b)
    nz_genes = sorted(np.nonzero(nzf)[0], key=lambda i: nzd[i])

    group_a = [gene_list[i] for i in nz_genes[: -(top_n + 1) : -1] if nzd[i] > 0]
    group_b = [gene_list[i] for i in nz_genes[:top_n] if nzd[i] < 0]

    return group_a, group_b


def collapse_tree(
    node_tree: NodeTree, sib_results: ResultsDict, min_de: int = 5, max_p: float = -10.0
):
    sib_totals = {k: get_de_count(sib_results[k], max_p=max_p) for k in sib_results}

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
