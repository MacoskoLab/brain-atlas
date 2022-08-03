import itertools
import logging
import pickle
import sys
from pathlib import Path

import click
import dask
import dask.array as da
import dask.distributed
import numba as nb
import numpy as np

import brain_atlas.find_genes as find_genes
from brain_atlas.util import create_logger

log = logging.getLogger("hail-de")


@click.command()
@click.option(
    "--input-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to pickle of DE triplets to compare",
)
@click.option(
    "--output-file",
    required=True,
    type=click.Path(path_type=Path),
    help="Path for output",
)
@click.option(
    "--cluster-data",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to cluster array npz",
)
@click.option("--array-path", required=True, help="Path to count zarr")
@click.option(
    "--start", required=True, help="Start index in comparison triple", type=int
)
@click.option(
    "--endexc",
    required=True,
    help="End index (exclusive) in comparison triple",
    type=int,
)
@click.option("--mem-limit", default="12GB", help="Dask mem limit")
@click.option("--n-workers", type=int, default=2, help="Number of dask workers to run")
@click.option(
    "--n-threads",
    type=int,
    default=2,
    help="Number of dask threads p/ worker to run",
)
@click.option("--n-subsample", type=int, default=40000)
@click.option("--debug", is_flag=True, help="Turn on debug logging")
def main(
    input_file: Path,
    output_file: Path,
    cluster_data: Path,
    array_path: str,
    start: int,
    endexc: int,
    mem_limit: str = "12GB",
    n_workers: int = 6,
    n_threads: int = 2,
    n_subsample: int = 40000,
    debug: bool = False,
):
    create_logger(debug)
    log.debug(f"Number of numba threads: {nb.get_num_threads()}")
    log.info(input_file)
    log.info(output_file)

    cluster = dask.distributed.LocalCluster(
        n_workers=n_workers,
        threads_per_worker=n_threads,
        # If increase hail mem size, increase here
        memory_limit=mem_limit,
    )
    client = dask.distributed.Client(cluster)
    count_array = da.from_zarr(array_path, "counts")

    # All these files are precalculated and loaded into root by hail, see `coach` file
    with np.load(cluster_data) as d:
        clusters = d["clusters"]
        cluster_nz_arr = d["cluster_nz_arr"]
        cluster_count_arr = d["cluster_count_arr"]

    # Since the comparison triplet-tuple is so small, not worth exporting one
    # pickle per job. Just export one and then have hail take a range to run (Can
    # just submit one hail job per run, but since startup takes time (docker image
    # loading + pickle loading), can be worth it to amortize cost by running a few
    # diffexp runs per job with nested parallelism)
    with open(input_file, "rb") as handle:
        allcompos_out = pickle.load(handle)
        log.debug("finished loading")

    this_comps = list(itertools.islice(allcompos_out, start, endexc))

    # Make node_tree just an index
    def this_comp_function(k, node_tree):
        return this_comps[k]

    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        de_results = dict()
        find_genes.generic_de(
            this_comp_function,
            count_array,
            clusters,
            # Fake node_tree, get's the k'th element of the
            # comparison list from this_comp_function
            node_tree=range(len(this_comps)),
            cluster_nz=cluster_nz_arr,
            cluster_counts=cluster_count_arr,
            de_results=de_results,
            subsample=n_subsample,
        )

    # Write to output path from hail so takes care of exporting to GCS without enlarging docker
    with open(output_file, "wb") as handle:
        pickle.dump(de_results, handle)
        log.debug("DONE")

    # Can be lingering dask threads which were keeping hail running even after
    # returned. But happily sys.exit forces all children to die
    client.shutdown()
    log.debug("sys exit")
    sys.exit(0)


if __name__ == "__main__":
    main()
