import logging
from pathlib import Path
from typing import Union

import dask.array as da
import numba as nb
import numpy as np
from numcodecs import Blosc

log = logging.getLogger(__name__)


@nb.njit(parallel=True, fastmath=True)
def translate_kng(node_subset: np.ndarray, kng: np.ndarray):
    """
    Subset a kNN graph to only the edges between the included nodes, filling
    in the rest with random edges inside the range. Can be used to initialize
    the creation of a new kNN for these nodes.

    :param node_subset: a boolean array specifying which nodes to include
    :param kng: the original k-neighbors graph to process into a new graph
    """

    n_c = node_subset.sum()
    nz = node_subset.nonzero()[0]
    cs = (~node_subset).cumsum()

    # values of -1 will be set to random neighbors by NNDescent
    new_kng = -1 * np.ones((n_c, kng.shape[1]), np.int32)

    for ii in nb.prange(n_c):
        i = nz[ii]
        j = 1
        for k in kng[i, :]:
            if node_subset[k]:
                new_kng[ii, j] = k - cs[k]
                j += 1

    return new_kng


@nb.njit(parallel=True, fastmath=True)
def kng_to_edgelist(kng: np.ndarray, knd: np.ndarray, min_weight: float = 0.0):
    """
    Convert a knn graph and distances into an array of unique edges with weights.
    Removes self-edges. Note: does *not* convert distances to similarity scores
    """
    n, m = kng.shape
    edges = np.vstack((np.repeat(np.arange(n).astype(np.int32), m), kng.flatten())).T
    weights = np.zeros(n * m, dtype=knd.dtype)

    for i in nb.prange(n):
        for jj, j in enumerate(kng[i, :]):
            if i < j:
                # this edge is fine
                weights[i * m + jj] = knd[i, jj]
            elif i > j:
                for k in kng[j, :]:
                    if i == k:
                        # this is already included on the other end
                        break
                else:
                    weights[i * m + jj] = knd[i, jj]

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(fastmath=True)
def cosine_similarity(u: np.ndarray, v: np.ndarray):
    """
    Compute the cosine similarity (not distance) of two vectors
    """
    m = u.shape[0]
    udotv = 0.0
    u_norm = 0.0
    v_norm = 0.0
    for i in range(m):
        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    return ratio


@nb.njit(parallel=True, fastmath=True)
def full_cosine_similarity(data: np.ndarray):
    """
    Computes all-by-all cosine similarities and returns the dense array
    """
    n = data.shape[0]
    dist = np.eye(n, dtype=np.float64)

    # computing similarity here so higher -> better
    for i in nb.prange(n - 1):
        for j in range(i + 1, n):
            dist[i, j] = cosine_similarity(data[i, :], data[j, :])
            dist[j, i] = dist[i, j]

    return dist


@nb.njit(parallel=True, fastmath=True)
def cosine_edgelist(data: np.ndarray, min_weight: float = 0.0):
    """
    Compute the all-by-all cosine similarity graph directly from data.

    This is faster than the approximate method, for smaller arrays
    """
    n = data.shape[0]
    dist = full_cosine_similarity(data)

    nc2 = n * (n - 1) // 2
    edges = np.empty((nc2, 2), dtype=np.int32)
    weights = np.zeros(nc2, dtype=np.float64)

    for i in nb.prange(n - 1):
        nic2 = (n - i) * (n - i - 1) // 2
        for j in range(i + 1, n):
            k = nc2 - nic2 + (j - i - 1)
            edges[k, 0] = i
            edges[k, 1] = j
            weights[k] = dist[i, j]

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(parallel=True, fastmath=True)
def k_cosine_edgelist(data: np.ndarray, k: int, min_weight: float = 0.0):
    """
    Creates a kNN edgelist by calculating all-by-all similarities first.
    For smaller n, this is faster than using the NNDescent algorithm,
    at the expense of temporarily higher memory usage
    """
    n = data.shape[0]
    dist = full_cosine_similarity(data)

    kng = np.zeros((n, k), dtype=np.int32)
    knd = np.zeros((n, k), dtype=np.float64)

    for i in nb.prange(n):
        edge_i = np.argsort(dist[i, :])[: -(k + 1) : -1]
        kng[i, :] = edge_i
        for jj, j in enumerate(edge_i):
            knd[i, jj] = dist[i, j]

    return kng_to_edgelist(kng, knd, min_weight)


@nb.njit(parallel=True, fastmath=True)
def k_jaccard_edgelist(data: np.ndarray, k: int, min_weight: float = 0.0):
    """
    Creates a Jaccard edgelist by calculating all-by-all similarities first.
    For smaller n, this is faster than using the NNDescent algorithm,
    at the expense of temporarily higher memory usage
    """
    n = data.shape[0]
    dist = full_cosine_similarity(data)

    kng = np.zeros((n, k), dtype=np.int32)

    for i in nb.prange(n):
        kng[i, :] = np.argsort(dist[i, :])[: -(k + 1) : -1]

    return kng_to_jaccard(kng, min_weight)


@nb.njit(parallel=True, fastmath=True)
def compute_mutual_edges(kng: np.ndarray, knd: np.ndarray, min_weight: float = 0.0):
    """
    Takes the knn graph and distances from pynndescent, computes unique mutual edges
    and converts from distance to edge weight (1 - distance). Removes self-edges
    """
    n, m = kng.shape
    edges = np.vstack((np.repeat(np.arange(n).astype(np.int32), m), kng.flatten())).T
    weights = np.zeros(n * m, dtype=knd.dtype)

    for i in nb.prange(n):
        for jj, j in enumerate(kng[i, :]):
            if j <= i:
                # this edge is already included, or a self-edge
                continue
            for k in kng[j, :]:
                if i == k:
                    weights[i * m + jj] = 1 - knd[i, jj]
                    break

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(parallel=True, fastmath=True)
def kng_to_jaccard(kng: np.ndarray, min_weight: float = 0.0):
    """
    Takes the knn graph and computes jaccard shared-nearest-neighbor edges and weights
    for all neighbors. Removes self-edges.
    """
    n, m = kng.shape
    edges = np.vstack((np.repeat(np.arange(n).astype(np.int32), m), kng.flatten())).T
    weights = np.zeros(n * m, dtype=np.float32)

    for i in nb.prange(n):
        kngs = set(kng[i, :])

        for jj, j in enumerate(kng[i, :]):
            # skip self-edges
            if i == j:
                continue

            overlap = 0
            skip = False
            for k in kng[j, :]:
                if i == k and j < i:
                    # this edge is already included
                    skip = True
                    break
                if k in kngs:
                    overlap += 1

            if not skip:
                d = overlap / (2 * m - overlap)
                weights[i * m + jj] = d

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


@nb.njit(parallel=True, fastmath=True)
def kng_to_full_jaccard(kng: np.ndarray, min_weight: float = 0.0):
    """
    Takes the knn graph and computes jaccard shared-nearest-neighbor edges and weights
    for all neighbors of neighbors, which approaches the complete SNN graph. Removes
    self-edges.
    """
    n, m = kng.shape

    # bitpack edges into one int64 so we can remove duplicates
    kng2 = np.zeros(n * m * m, dtype=np.int64)

    for i in nb.prange(n):
        # because we have a self-edge, this includes the first-order edges
        js = np.unique(kng[kng[i, :], :])
        js_0 = (js[js < i] << 32) | i
        js_1 = (i << 32) | js[js > i]
        js = np.hstack((js_0, js_1))

        kng2[i * m * m : i * m * m + js.shape[0]] = js

    # remove duplicates
    kng2 = np.unique(kng2[kng2 > 0])

    # unpack back into edges
    edges = np.empty((kng2.shape[0], 2), dtype=np.int32)
    edges[:, 0] = kng2 >> 32
    edges[:, 1] = kng2 & 0xFFFFFFFF

    weights = np.empty(edges.shape[0], dtype=np.float32)

    for ii in nb.prange(edges.shape[0]):
        i = edges[ii, 0]
        j = edges[ii, 1]

        overlap = 0
        for v_i in kng[i, :]:
            for v_j in kng[j, :]:
                if v_i == v_j:
                    overlap += 1

        d = overlap / (2 * m - overlap)
        weights[ii] = d

    ix = weights > min_weight
    return edges[ix, :], weights[ix]


def write_knn_to_zarr(
    kng: np.ndarray,
    knd: np.ndarray,
    zarr_path: Union[str, Path],
    chunk_rows: int = 100000,
    overwrite: bool = False,
):
    """
    Writes kNN graph and distances to disk as zarr files
    """
    log.debug(f"Writing neighbor graph to {zarr_path}/kng")
    da.array(kng.astype(np.int32)).rechunk((chunk_rows, kng.shape[1])).to_zarr(
        zarr_path,
        "kng",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
    log.debug(f"Writing edge distances to {zarr_path}/knd")
    da.array(knd).rechunk((chunk_rows, knd.shape[1])).to_zarr(
        zarr_path,
        "knd",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )


def write_edges_to_zarr(
    edges: np.ndarray,
    weights: np.ndarray,
    zarr_path: Union[str, Path],
    chunk_rows: int = 100000,
    overwrite: bool = False,
):
    """
    Writes edge and weight arrays to disk as zarr files
    """
    log.debug(f"Writing edges to {zarr_path}/edges")
    da.array(edges.astype(np.int32)).rechunk((chunk_rows, 2)).to_zarr(
        zarr_path,
        "edges",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
    log.debug(f"Writing edge weights to {zarr_path}/weights")
    da.array(weights).rechunk((chunk_rows,)).to_zarr(
        zarr_path,
        "weights",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
