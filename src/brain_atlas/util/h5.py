import dask
import h5py
import numpy as np
import scipy.sparse
import sparse
from gcsfs import GCSFileSystem


def read_10x_h5(path: str):
    with h5py.File(path, "r") as fh:
        M, N = fh["matrix"]["shape"]
        gene_names = tuple(fh["matrix"]["features"]["name"].asstr())
        gene_ids = tuple(fh["matrix"]["features"]["id"].asstr())

        genes = tuple(zip(gene_names, gene_ids))
        barcodes = tuple(fh["matrix"]["barcodes"].asstr())

        data = np.asarray(fh["matrix"]["data"])
        indices = np.asarray(fh["matrix"]["indices"])
        indptr = np.asarray(fh["matrix"]["indptr"])

    matrix = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N, M))

    return matrix, barcodes, genes


@dask.delayed(nout=3, pure=True)
def read_10x_h5_meta_from_gcs(path: str, fs: GCSFileSystem):
    lib_name = path.rsplit("/", 1)[-1].split("_")[0]

    with fs.open(path, "rb") as gcs_fh:
        with h5py.File(gcs_fh, "r") as fh:
            M, N = fh["matrix"]["shape"]
            gene_names = tuple(fh["matrix"]["features"]["name"].asstr())
            gene_ids = tuple(fh["matrix"]["features"]["id"].asstr())

            genes = tuple(zip(gene_names, gene_ids))
            barcodes = tuple(fh["matrix"]["barcodes"].asstr())

    barcodes = [f"{lib_name}_{c}" for c in barcodes]
    return (N, M), (genes,), barcodes


@dask.delayed(pure=True, nout=1)
def read_10x_numis_from_gcs(path: str, fs: GCSFileSystem):
    with fs.open(path, "rb") as gcs_fh:
        with h5py.File(gcs_fh, "r") as fh:
            M, N = fh["matrix"]["shape"]
            data = np.asarray(fh["matrix"]["data"])
            indices = np.asarray(fh["matrix"]["indices"])
            indptr = np.asarray(fh["matrix"]["indptr"])

    numis = (
        sparse.GCXS((data, indices, indptr), shape=(N, M), compressed_axes=(0,))
        .sum(1)
        .todense()
    )

    return numis


@dask.delayed(pure=True, nout=1)
def get_10x_mt(path: str, numis: np.ndarray, mt_ix: np.ndarray, fs: GCSFileSystem):
    with fs.open(path, "rb") as gcs_fh:
        with h5py.File(gcs_fh, "r") as fh:
            M, N = fh["matrix"]["shape"]
            data = np.asarray(fh["matrix"]["data"])
            indices = np.asarray(fh["matrix"]["indices"])
            indptr = np.asarray(fh["matrix"]["indptr"])

    matrix = sparse.GCXS((data, indices, indptr), shape=(N, M), compressed_axes=(0,))
    mt_pct = (matrix[:, mt_ix].sum(1) / np.maximum(numis, 1)).todense()

    return mt_pct


@dask.delayed(pure=True, nout=1)
def read_10x_h5_from_gcs(path: str, cell_filter: np.ndarray, fs: GCSFileSystem):
    with fs.open(path, "rb") as gcs_fh:
        with h5py.File(gcs_fh, "r") as fh:
            M, N = fh["matrix"]["shape"]
            data = np.asarray(fh["matrix"]["data"])
            indices = np.asarray(fh["matrix"]["indices"])
            indptr = np.asarray(fh["matrix"]["indptr"])

    matrix = sparse.GCXS(
        (data, indices, indptr),
        shape=(N, M),
        compressed_axes=(0,),
    )[cell_filter, :].todense()

    return matrix
