import logging
from pathlib import Path

import click
import numpy as np
import zarr
from numcodecs import Blosc

from brain_atlas.util import optional_gzip
from brain_atlas.util.dataset import Dataset
from brain_atlas.util.h5 import read_10x_h5

log = logging.getLogger(__name__)


@click.command(name="make-zarr", no_args_is_help=True)
@click.argument("h5-files", nargs=-1, metavar="H5_FILE [H5_FILE ... ]")
@click.option(
    "--output-zarr", required=True, type=click.Path(dir_okay=True, file_okay=False)
)
@click.option("--output-cells", type=click.Path(), help="Path to write cell barcodes")
@click.option(
    "--output-genes", type=click.Path(), help="Path to write genes and gene ids"
)
@click.option(
    "--min-umis", type=int, default=500, help="Minimum number of UMIs per cell"
)
def main(
    h5_files: str,
    output_zarr: str,
    output_cells: str = None,
    output_genes: str = None,
    min_umis: int = 500,
):
    """Converts a list of H5_FILE into a Zarr array. Only cells above MIN-UMIS are kept.
    Cell names will be made by concatenating the library name to the cell barcode. If
    the h5 files do not all have the same gene list, an error is raised.
    """
    h5_files = sorted(Path(fp) for fp in h5_files)

    log.info(f"reading {len(h5_files)} h5 files")

    cell_lists = []
    gene_set = set()

    log.info(f"writing output to {output_zarr}")
    g = zarr.open_group(store=output_zarr, mode="w")

    z = g.create_dataset(
        name=Dataset.COUNTS,
        shape=(0, 0),
        chunks=Dataset.CHUNKS,
        dtype=np.int32,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )

    z_u = g.create_dataset(
        name=Dataset.NUMIS,
        shape=(0, 1),
        chunks=(Dataset.CHUNKS, 1),
        dtype=np.int64,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )

    for fp in h5_files:
        i, j = z.shape
        m, mcells, mgenes = read_10x_h5(str(fp))
        assert len(mcells) == len(set(mcells))
        assert j == 0 or j == m.shape[1]

        gene_set.add(mgenes)

        numis = np.asarray(m.sum(1)).squeeze()
        over_min = numis >= min_umis
        m = m[over_min, :].tocoo()

        z.resize(i + m.shape[0], m.shape[1])
        z.set_coordinate_selection((m.row + i, m.col), m.data)
        cell_lists.append([c for c, b in zip(mcells, over_min) if b])

        z_u.resize(i + m.shape[0], 1)
        z_u[i:, 0] = numis[over_min]

    log.info(f"Wrote a total of {z.shape[0]} cells")

    assert len(gene_set) == 1, "Multiple gene lists found"
    genes = gene_set.pop()

    if output_cells is not None:
        log.info(f"Writing cell list to {output_cells}")
        with optional_gzip(output_cells, "w") as out:
            for h5_fp, cl in zip(h5_files, cell_lists):
                lib = h5_fp.name.split("_", 1)[0]
                for c in cl:
                    print(f"{lib}_{c}", file=out)

    if output_genes is not None:
        log.info(f"Writing gene list to {output_genes}")
        with optional_gzip(output_genes, "w") as out:
            print("\n".join(f"{gene}\t{gid}" for gene, gid in genes), file=out)

    log.info("Done!")
