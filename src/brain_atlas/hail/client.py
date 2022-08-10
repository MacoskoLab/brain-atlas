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
    help="Path to node tree pickle on GCS",
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
@click.option("--max-nz-b", type=float, default=0.2)
@click.option("--delta-nz", type=float, default=0.2)
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
    max_nz_b: float = 0.2,
    delta_nz: float = 0.2,
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

    # this file is precalculated and loaded into root by hail
    with np.load(cluster_data) as d:
        clusters = d["clusters"]
        cluster_nz_arr = d["cluster_nz_arr"]
        cluster_count_arr = d["cluster_count_arr"]

    log.debug(f"Loading node_tree from {input_file}")
    with open(input_file, "rb") as fh:
        node_tree = pickle.load(fh)
        log.debug("finished loading")

    # use islice to grab a small chunk of comparison keys without computing indices
    comp_list = list(
        itertools.islice(
            (
                (k1, k2)
                for k1, k2 in itertools.combinations(sorted(node_tree), 2)
                if k1 != k2[: len(k1)]
            ),
            start,
            endexc,
        )
    )

    # only compute cluster indices for the small chunk
    comp_list = [
        (
            (k1, k2),
            np.array(node_tree[k1].pre_order(True, lambda nd: nd.index)),
            np.array(node_tree[k2].pre_order(True, lambda nd: nd.index)),
        )
        for k1, k2 in comp_list
    ]

    de_results = dict()
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        # same logic as find_genes.generic_de, somewhat simplified
        for comp, c_i, c_j in comp_list:
            log.debug(f"Comparing {c_i} with {c_j}")
            nz_i, nz_j, nz_filter = find_genes.calc_filter(
                cluster_count_arr, cluster_nz_arr, c_i, c_j, max_nz_b, delta_nz
            )

            _, p = find_genes.de(
                count_array, clusters, c_i, c_j, nz_filter, n_subsample
            )
            de_results[comp] = p, nz_i, nz_j, nz_filter

    # Write to output path from hail so takes care of exporting to GCS without enlarging docker
    with open(output_file, "wb") as handle:
        pickle.dump(de_results, handle)
        log.debug("Done!")

    # Can be lingering dask threads which were keeping hail running even after
    # returned. But happily sys.exit forces all children to die
    client.shutdown()
    log.debug("sys exit")
    sys.exit(0)


if __name__ == "__main__":
    main()
