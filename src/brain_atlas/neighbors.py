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

    new_kng = np.random.randint(0, n_c, size=(n_c, kng.shape[1]))

    for ii in nb.prange(n_c):
        i = nz[ii]
        j = 0
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


@nb.njit(parallel=True)
def compute_jaccard_edges(kng: np.ndarray, min_dist=0.0625):
    n, m = kng.shape
    dists = np.vstack((np.repeat(np.arange(n), m), kng.flatten(), np.zeros(n * m))).T

    for i in nb.prange(n):
        kngs = set(kng[i, :])

        for jj, j in enumerate(kng[i, :]):
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
                dists[i * m + jj, 2] = d

    return dists[dists[:, 2] > min_dist, :]


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
        zarr_path,
        "edges",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
    log.debug(f"Writing edge weights to {zarr_path}/weights")
    da.array(jaccard_edge_array[:, 2]).rechunk((chunk_rows,)).to_zarr(
        zarr_path,
        "weights",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
