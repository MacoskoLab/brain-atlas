import logging

import click
import dask
import dask.array as da
import numpy as np
from dask.distributed import Client
from numcodecs import Blosc

from ..util import optional_gzip

log = logging.getLogger(__name__)


@click.command(name="filter_mt", no_args_is_help=True)
@click.argument("input-zarr", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("output-zarr", type=click.Path(dir_okay=True, file_okay=False))
@click.option("-d", "--dask-client", required=True)
@click.option("-g", "--genes", required=True, type=click.Path(exists=True))
@click.option("-c", "--cells", required=True, type=click.Path(exists=True))
@click.option("-o", "--output-cells", required=True, type=click.Path())
@click.option("-p", "--max-pct", type=float, default=0.01)
def main(
    input_zarr: str,
    output_zarr: str,
    dask_client: str,
    genes: str,
    cells: str,
    output_cells: str = None,
    max_pct: float = 0.01,
):
    client = Client(dask_client)
    log.debug(f"connected to client {client}")

    log.info(f"Reading genes from {genes}")
    with optional_gzip(genes, "r") as fh:
        gene_list, gene_ids = zip(*[line.strip().split() for line in fh])

    log.debug(f"Read {len(gene_list)} genes from {genes}")

    mito_idx = [i for i, g in enumerate(gene_list) if g.startswith("mt")]
    log.debug(f"Found {len(mito_idx)} genes starting with 'mt'")

    log.info(f"Reading cell list from {cells}")
    with optional_gzip(cells, "r") as fh:
        input_cells = [line.strip() for line in fh]

    log.debug(f"Read {len(input_cells)} cells from {cells}")

    log.debug(f"Reading from {input_zarr}")
    ds = da.from_zarr(input_zarr)

    numis = ds.sum(1)
    mdist = ds[:, mito_idx].sum(1)

    log.info("Computing mitochondrial ratio")
    m_ratio = (mdist / numis).compute()

    log.info(
        "Mitochondrial distribution:\n"
        f"\t{np.percentile(m_ratio, (20, 50, 80, 97.5, 98, 99))}"
    )

    log.info(f"Filtering out cells with >= {max_pct:.0%} mito reads")
    log.debug(f"Writing output to {output_zarr}")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        ds[m_ratio < max_pct, :].rechunk((4000, 4000)).to_zarr(
            output_zarr,
            compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
        )

    log.debug(f"Writing filtered cell list to {output_cells}")
    with optional_gzip(output_cells, "w") as out:
        for c, b in zip(input_cells, m_ratio < max_pct):
            if b:
                print(c, file=out)

    log.info("Done!")
