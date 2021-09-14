from os import PathLike

import dask.array as da
import numba as nb
import numpy as np
from numcodecs import Blosc


@nb.njit(parallel=True)
def compute_jaccard_edges(kng: np.array, min_dist=0.0625):
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
    jaccard_edge_array: np.array,
    zarr_path: PathLike[str],
    chunk_rows: int = 100000,
    overwrite: bool = False,
):
    da.array(jaccard_edge_array[:, :2].astype(np.int32)).rechunk(
        (chunk_rows, 2)
    ).to_zarr(
        zarr_path,
        "edges",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
    da.array(jaccard_edge_array[:, 2]).rechunk((chunk_rows,)).to_zarr(
        zarr_path,
        "weights",
        overwrite=overwrite,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
