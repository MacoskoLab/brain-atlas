import logging
from os import PathLike

import dask.array as da
import numba as nb
import numpy as np
from numcodecs import Blosc

log = logging.getLogger(__name__)


@nb.njit(parallel=True)
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


def write_knn_to_zarr(
    kng: np.ndarray,
    knd: np.ndarray,
    zarr_path: PathLike,
    chunk_rows: int = 100000,
    overwrite: bool = False,
):
    log.debug(f"Writing neighbor graph to {zarr_path}/kng")
    da.array(kng.astype(np.int32)).rechunk((chunk_rows, kng.shape[1])).to_zarr(
        str(zarr_path),
        "kng",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
    log.debug(f"Writing edge distances to {zarr_path}/knd")
    da.array(knd).rechunk((chunk_rows, knd.shape[1])).to_zarr(
        str(zarr_path),
        "knd",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )


@nb.njit(parallel=True)
def kng_to_edgelist(kng: np.ndarray, knd: np.ndarray):
    """
    Convert a knn graph and distances into an array of unique edges with weights.
    Removes self-edges
    """
    n, m = kng.shape
    edges = np.vstack((np.repeat(np.arange(n), m), kng.flatten(), np.zeros(n * m))).T

    for i in nb.prange(n):
        for jj, j in enumerate(kng[i, :]):
            if i < j:
                # this edge is fine
                edges[i * m + jj, 2] = 1 - knd[i, jj]
            elif i > j:
                for k in kng[j, :]:
                    if i == k:
                        # this is already included on the other end
                        break
                else:
                    edges[i * m + jj, 2] = 1 - knd[i, jj]

    return edges[edges[:, 2] > 0, :]


@nb.njit
def cosine_similarity(u: np.ndarray, v: np.ndarray):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

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


@nb.njit(parallel=True)
def cosine_edgelist(data: np.ndarray):
    """
    Compute the all-by-all cosine similarity graph directly from data.

    This is faster than the approximate method, for smaller arrays
    """
    n = data.shape[0]
    nc2 = n * (n - 1) // 2
    edges = np.empty((nc2, 3), dtype=np.float64)

    for i in nb.prange(n - 1):
        nic2 = (n - i) * (n - i - 1) // 2
        for j in range(i + 1, n):
            ix = nc2 - nic2 + (j - i - 1)
            edges[ix, 0] = i
            edges[ix, 1] = j
            edges[ix, 2] = cosine_similarity(data[i, :], data[j, :])
    return edges


@nb.njit(parallel=True)
def compute_mutual_edges(kng: np.ndarray, knd: np.ndarray):
    """
    Takes the knn graph and distances from pynndescent, computes unique mutual edges
    and converts from distance to edge weight (1 - distance). Removes self-edges
    """
    n, m = kng.shape
    edges = np.vstack((np.repeat(np.arange(n), m), kng.flatten(), np.zeros(n * m))).T

    for i in nb.prange(n):
        for jj, j in enumerate(kng[i, :]):
            if j <= i:
                # this edge is already included, or a self-edge
                continue
            for k in kng[j, :]:
                if i == k:
                    edges[i * m + jj, 2] = 1 - knd[i, jj]
                    break

    return edges[edges[:, 2] > 0, :]


@nb.njit(parallel=True)
def compute_jaccard_edges(kng: np.ndarray, min_dist=0.0625):
    """
    Takes the knn graph and computes jaccard shared-nearest-neighbor edges and weights
    """
    n, m = kng.shape
    edges = np.vstack((np.repeat(np.arange(n), m), kng.flatten(), np.zeros(n * m))).T

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
                edges[i * m + jj, 2] = d

    return edges[edges[:, 2] > min_dist, :]


def write_jaccard_to_zarr(
    jaccard_edge_array: np.ndarray,
    zarr_path: PathLike,
    chunk_rows: int = 100000,
    overwrite: bool = False,
):
    log.debug(f"Writing jaccard edges to {zarr_path}/edges")
    da.array(jaccard_edge_array[:, :2].astype(np.int32)).rechunk(
        (chunk_rows, 2)
    ).to_zarr(
        str(zarr_path),
        "edges",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
    log.debug(f"Writing edge weights to {zarr_path}/weights")
    da.array(jaccard_edge_array[:, 2]).rechunk((chunk_rows,)).to_zarr(
        str(zarr_path),
        "weights",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
