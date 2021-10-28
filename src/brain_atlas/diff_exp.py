import numba as nb
import numpy as np
import scipy.stats


@nb.njit(parallel=True)
def tiecorrect(rankvals):
    """
    parallelized version of scipy.stats.tiecorrect

    :param rankvals: p x n array of ranked data (output of rankdata function)
    """
    tc = np.ones(rankvals.shape[1], dtype=np.float64)
    for j in nb.prange(rankvals.shape[1]):
        arr = np.sort(np.ravel(rankvals[:, j]))
        idx = np.nonzero(
            np.concatenate((np.array([True]), arr[1:] != arr[:-1], np.array([True])))
        )[0]
        t_k = np.diff(idx).astype(np.float64)

        size = np.float64(arr.size)
        if size >= 2:
            tc[j] = 1.0 - (t_k ** 3 - t_k).sum() / (size ** 3 - size)

    return tc


@nb.njit(parallel=True)
def rankdata(data):
    """
    parallelized version of scipy.stats.rankdata

    :param data: p x n array of data to rank, column-wise
    """
    ranked = np.empty(data.shape, dtype=np.float64)
    for j in nb.prange(data.shape[1]):
        arr = np.ravel(data[:, j])
        sorter = np.argsort(arr)

        arr = arr[sorter]
        obs = np.concatenate((np.array([True]), arr[1:] != arr[:-1]))

        dense = np.empty(obs.size, dtype=np.int64)
        dense[sorter] = obs.cumsum()

        # cumulative counts of each unique value
        count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))
        ranked[:, j] = 0.5 * (count[dense] + count[dense - 1] + 1)

    return ranked


def mannwhitneyu(x, y, use_continuity=True):
    """Version of Mann-Whitney U-test that runs in parallel on 2d arrays

    This is the two-sided test, asymptotic algo only. Returns log p-values
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert x.shape[1] == y.shape[1]

    n1 = x.shape[0]
    n2 = y.shape[0]

    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[:n1, :]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1 * n2 - u1  # remainder is U for y
    T = tiecorrect(ranked)

    # if *everything* is identical we'll raise an error, not otherwise
    if np.all(T == 0):
        raise ValueError("All numbers are identical in mannwhitneyu")
    sd = np.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)

    meanrank = n1 * n2 / 2.0 + 0.5 * use_continuity
    bigu = np.maximum(u1, u2)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = (bigu - meanrank) / sd

    logp = np.clip(2 * scipy.stats.norm.logsf(z), 0, 1)

    return u2, logp
