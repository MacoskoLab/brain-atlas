import logging
from pathlib import Path

import click
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm.auto import tqdm

from ..util import optional_gzip
from ..util.h5 import read_10x_h5

log = logging.getLogger(__name__)


@click.command(name="make_zarr", no_args_is_help=True)
@click.argument("h5_files", nargs=-1, metavar="H5_FILE [H5_FILE ... ]")
@click.option(
    "--output-zarr", required=True, type=click.Path(dir_okay=True, file_okay=False)
)
@click.option("--output-cells", type=click.Path())
@click.option("--output-genes", type=click.Path())
@click.option("--min-umis", type=int, default=500)
def main(
    h5_files: list[str],
    output_zarr: str,
    output_cells: str = None,
    output_genes: str = None,
    min_umis: int = 500,
):
    h5_files = [Path(fp) for fp in sorted(h5_files)]

    log.info(f"reading {len(h5_files)} h5 files")

    x, cells, genes, gids = read_10x_h5(h5_files[0])

    log.debug(f"Read {len(genes)} genes")

    log.debug(f"filtering to >= {min_umis} UMIs")
    over_min = np.asarray(x.sum(1) >= min_umis).squeeze()

    cell_lists = [[c for c, b in zip(cells, over_min) if b]]
    gene_set = {genes}
    gid_set = {gids}

    x = x[over_min, :].tocoo()

    log.info(f"writing output to {output_zarr}")
    z = zarr.create(
        store=output_zarr,
        shape=x.shape,
        chunks=(4000, 4000),
        dtype=x.dtype,
        compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
    )
    z.set_coordinate_selection((x.row, x.col), x.data)

    for fp in tqdm(h5_files[1:]):
        i, j = z.shape
        m, mcells, mgenes, mgids = read_10x_h5(fp)
        assert len(mcells) == len(set(mcells))

        gene_set.add(mgenes)
        gid_set.add(mgids)

        over_min = np.asarray(m.sum(1) >= min_umis).squeeze()
        m = m[over_min, :].tocoo()

        z.resize(i + m.shape[0], j)
        z.set_coordinate_selection((m.row + i, m.col), m.data)
        cell_lists.append([c for c, b in zip(mcells, over_min) if b])

    log.info(f"Wrote a total of {z.shape[0]} cells")

    assert len(gene_set) == 1
    genes = gene_set.pop()
    assert len(gid_set) == 1
    gids = gid_set.pop()

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
            print(
                "\n".join(f"{gene}\t{gid}" for gene, gid in zip(genes, gids)), file=out
            )

    log.info("Done!")
