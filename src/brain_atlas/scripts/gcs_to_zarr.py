import logging

import click
import dask
import dask.array as da
import gcsfs
import numpy as np
from numcodecs import Blosc

from brain_atlas.util import optional_gzip
from brain_atlas.util.dataset import Dataset
from brain_atlas.util.h5 import (
    get_10x_mt,
    read_10x_h5_from_gcs,
    read_10x_h5_meta_from_gcs,
    read_10x_numis_from_gcs,
)

log = logging.getLogger(__name__)


@click.command(name="make-zarr", no_args_is_help=True)
@click.argument("h5-files", nargs=-1, metavar="H5_FILE [H5_FILE ... ]")
@click.option(
    "--output-zarr", required=True, type=click.Path(dir_okay=True, file_okay=False)
)
@click.option("--output-cells", required=True, type=click.Path())
@click.option("--output-genes", required=True, type=click.Path())
@click.option("--min-umis", type=int, default=500)
@click.option("-p", "--max-pct", type=float, default=0.01)
@click.option("--google-project", help="GCP Project ID to use")
def main(
    h5_files: str,
    output_zarr: str,
    output_cells: str = None,
    output_genes: str = None,
    min_umis: int = 500,
    max_pct: float = 0.01,
    google_project: str = None,
):
    """Downloads a list of H5_FILE from Google Cloud Storage and writes them to Zarr.
    Filters to cells with at least MIN-UMIS and less than MAX-PCT mitochondrial UMIs.

    Cell names will be made by concatenating the library name to the cell barcode. If
    the h5 files do not all have the same gene list, an error is raised.
    """

    fs = gcsfs.GCSFileSystem(project=google_project)

    log.info(f"reading metadata for {len(h5_files)} h5 files")

    sgb = [read_10x_h5_meta_from_gcs(path, fs) for path in h5_files]
    sizes, genes, barcodes = zip(*dask.compute(sgb)[0])

    assert len(genes) == 1, "Multiple gene lists found"
    genes = genes.pop()
    mito_idx = np.array(
        [i for i, (gene_name, _) in enumerate(genes) if gene_name.startswith("mt")]
    )

    log.info("Calculating nUMIs and mito pct")
    numis = [read_10x_numis_from_gcs(path, fs) for path in h5_files]
    mito_pct = [get_10x_mt(path, n, mito_idx, fs) for n, path in zip(numis, h5_files)]

    h5_filters = [((n >= min_umis) & (mt < max_pct)) for n, mt in zip(numis, mito_pct)]
    h5_filters, numis, mito_pct = da.compute(h5_filters, numis, mito_pct)

    new_barcodes = [
        bc
        for bc_list, h5f in zip(barcodes, h5_filters)
        for bc, b in zip(bc_list, h5f)
        if b
    ]
    new_sizes = [h5f.sum() for h5f in h5_filters]

    big_numis = np.vstack([n[h5f, None] for n, h5f in zip(numis, h5_filters)])
    da.array(big_numis).rechunk((Dataset.CHUNKS, 1)).to_zarr(
        output_zarr,
        Dataset.NUMIS,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )

    log.info("Reading data and writing to Zarr")
    matrices = [
        read_10x_h5_from_gcs(path, cf) for path, cf in zip(h5_files, h5_filters)
    ]

    big_array = da.vstack(
        [
            da.from_delayed(m, shape=(s, len(genes)), dtype=np.uint32)
            for m, s in zip(matrices, new_sizes)
        ]
    ).rechunk(Dataset.CHUNKS)

    big_array.to_zarr(
        output_zarr,
        Dataset.COUNTS,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )

    log.info(f"Writing cell list to {output_cells}")
    with optional_gzip(output_cells, "w") as out:
        print("\n".join(new_barcodes), file=out)

    log.info(f"Writing gene list to {output_genes}")
    with optional_gzip(output_genes, "w") as out:
        print("\n".join(f"{gene}\t{gid}" for gene, gid in genes), file=out)

    log.info("Done!")
