import logging
from typing import Union

import dask.array as da
import numpy as np
import scipy.stats

log = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, da.Array]


# blockwise poisson fit of gene counts
def dask_pblock(counts: ArrayLike, numis: ArrayLike = None, blocksize: int = 128000):
    n_cells = counts.shape[0]

    # pre-compute these values
    log.debug("computing percent nonzero per gene")
    pct = da.compute((counts > 0).sum(0) / n_cells)[0]

    log.debug("computing average expression per gene")
    exp = counts.sum(0, keepdims=True)
    exp = exp / exp.sum()

    if numis is None:
        numis = counts.sum(1, keepdims=True)

    exp_pct_nz = np.zeros(exp.shape)  # 1 x n_genes
    var_pct_nz = np.zeros(exp.shape)  # 1 x n_genes

    log.debug("computing expected percent nonzero")
    # run in chunks (still large, but seems easier for dask to handle)
    for i in range(0, n_cells, blocksize):
        if i % (blocksize * 10) == 0:
            log.debug(f"{i} ...")

        prob_zero = np.exp(-exp.T.dot(numis[i : i + blocksize, :].T))  # n_genes x b

        exp_pct_nz_b = (1 - prob_zero).sum(1)  # n_genes
        var_pct_nz_b = (prob_zero * (1 - prob_zero)).sum(1)

        exp_pct_nz_b, var_pct_nz_b = da.compute(exp_pct_nz_b, var_pct_nz_b)

        exp_pct_nz += exp_pct_nz_b
        var_pct_nz += var_pct_nz_b

    exp_pct_nz = exp_pct_nz.squeeze() / n_cells
    std_pct_nz = np.sqrt(var_pct_nz.squeeze()) / n_cells

    log.debug("... done")

    exp_p = np.zeros_like(pct)
    ix = (std_pct_nz != 0).flatten()
    exp_p[ix] = scipy.stats.norm.logcdf(
        pct[ix], loc=exp_pct_nz[ix], scale=std_pct_nz[ix]
    )

    return exp_pct_nz, pct, exp_p
