import logging

import dask.array as da
from numcodecs import Blosc

log = logging.getLogger(__name__)


class Dataset:
    COUNTS = "counts"
    NUMIS = "numis"
    CHUNKS = 4000

    def __init__(
        self, input_zarr: str, count_array: str = COUNTS, numi_array: str = NUMIS
    ):
        self.counts = da.from_zarr(input_zarr, count_array)
        self.numis = da.from_zarr(input_zarr, numi_array)

    @staticmethod
    def save(output_zarr, count_array: da.Array, numis: da.Array = None):
        log.debug(f"saving to {output_zarr}")
        count_array = count_array.rechunk(Dataset.CHUNKS)
        count_array.to_zarr(
            output_zarr,
            Dataset.COUNTS,
            compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
        )

        if numis is None:
            log.warning("Computing numis per cell")
            numis = count_array.sum(1, keepdims=True)

        numis = numis.rechunk((Dataset.CHUNKS, 1))
        numis.to_zarr(
            output_zarr,
            Dataset.NUMIS,
            compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
        )
