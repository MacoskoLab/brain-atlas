import logging

import dask.array as da
import numpy as np
import scipy.stats

log = logging.getLogger(__name__)


# blockwise poisson fit of gene counts
def dask_pblock(ds: da.Array, numis: da.Array = None, b: int = 128000):
    n_cells = ds.shape[0]

    # pre-compute these values
    log.debug("computing percent nonzero per gene")
    pct = ((ds > 0).sum(0) / n_cells).compute()
    log.debug("persisting average expression per gene")
    exp = ds.sum(0, keepdims=True)
    exp = (exp / exp.sum()).persist()
    if numis is None:
        numis = ds.sum(1, keepdims=True)

    exp_pct_nz = np.zeros(exp.shape)  # 1 x n_genes
    var_pct_nz = np.zeros(exp.shape)  # 1 x n_genes

    log.debug("computing expected percent nonzero")
    # run in chunks (still large, but seems easier for dask to handle)
    for i in range(0, n_cells, b):
        if i % (b * 10) == 0:
            log.debug(f"{i} ...")

        prob_zero = np.exp(-exp.T.dot(numis[i : i + b, :].T))  # n_genes x b

        exp_pct_nz_b = (1 - prob_zero).sum(1)  # n_genes
        var_pct_nz_b = (prob_zero * (1 - prob_zero)).sum(1)

        exp_pct_nz_b, var_pct_nz_b = da.compute(exp_pct_nz_b, var_pct_nz_b)

        exp_pct_nz += exp_pct_nz_b
        var_pct_nz += var_pct_nz_b

    exp_pct_nz = exp_pct_nz.flatten() / n_cells
    var_pct_nz = var_pct_nz.flatten() / (n_cells * n_cells)
    std_pct_nz = np.sqrt(var_pct_nz)

    log.debug("... done")

    exp_p = np.zeros_like(pct)
    ix = (std_pct_nz != 0).flatten()
    exp_p[ix] = scipy.stats.norm.logcdf(
        pct[ix], loc=exp_pct_nz[ix], scale=std_pct_nz[ix]
    )

    return exp_pct_nz, pct, exp_p
