import gzip
import itertools
import pickle
import sys
import time

import click
import dask
import dask.array as da
import dask.distributed
import numba as nb

import brain_atlas.find_genes as find_genes

# FILES NEED
# - atlas_500umis_mt-1pct_cells.txt.gz
# - gene_list.txt
# - 01_mtx_zarr/counts
# - 01_mtx_zarr/atlas_500umis_mt-1pct_merged_clusters.zarr
# - 01_mtx_zarr/atlas_500umis_mt-1pct_cells.txt.gz
# gs://macosko_data/jlanglie/hail_MBASS_files/


# If running on local docker for debugging, helpful for copying
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/atlas_500umis_mt-1pct_cells.txt.gz"  25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/gene_list.txt"                       25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/clusters.pickle"             25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/cluster_nz_arr.pickle"       25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/cluster_count_arr.pickle"    25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/hail_de_runner.py"      25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/tmpPickleOut.pickle"         25ae21375756:/


@click.command()
@click.option("--input-file", required=True, help="Location of input pickle")
@click.option("--output-file", required=True, help="Location of output pickle")
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
def main(
    input_file,
    output_file,
    start,
    endexc,
    mem_limit="12GB",
    n_workers=6,
    n_threads=2,
    n_subsample=40000,
):
    print(f"Number of numba threads: {nb.get_num_threads()}")
    print(input_file)
    print(output_file)

    # array_path = "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/counts"
    # cluster_path = "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/atlas_500umis_mt-1pct_merged_clusters.zarr"
    array_path = "gs://macosko_data/jlanglie/hail_MBASS_files/01_mtx_zarr/counts"
    cluster_path = "gs://macosko_data/jlanglie/hail_MBASS_files/01_mtx_zarr/atlas_500umis_mt-1pct_merged_clusters.zarr"

    cluster = dask.distributed.LocalCluster(
        n_workers=n_workers,
        threads_per_worker=n_threads,
        # If increase hail mem size, increase here
        memory_limit=mem_limit,
    )
    client = dask.distributed.Client(cluster)

    cluster_array = da.from_zarr(cluster_path)
    count_array = da.from_zarr(array_path)

    # compute the cluster_array (not too big to keep in RAM)
    # TODO for hail, maybe worth keeping on disk but probably not.
    # Maybe faster to just precalc a pickle instead of going through dask
    cluster_array = cluster_array.compute()

    # Just for Jonah but leave in please. Helps me keep track of different
    # clustering versions so I don't mix derived files in later mappings in case
    # of key collision
    forJonah = False
    if forJonah:
        # For future proofing this against future clustering generations
        # Prepend generation to all keys, so make sure changes everything globally
        CLUSTER_GENERATION = 10
        # As an integer, do CLUSTER_GENERATION*10 + FIRST ELEMENT
        # As a string, just add on str(CLUSTER_GENERATION)+str(FIRST ELEMENT)
        cluster_array[:, 0] += CLUSTER_GENERATION * 10

    # All these files are precalculated and loaded into root by hail, see `coach` file
    cluster_file = "/clusters.pickle"
    cluster_nz_arr_file = "/cluster_nz_arr.pickle"
    cluster_count_arr_file = "/cluster_count_arr.pickle"

    with open(cluster_file, "rb") as fn:
        clusters = pickle.load(fn)

    with open(cluster_nz_arr_file, "rb") as fn:
        cluster_nz_arr = pickle.load(fn)

    with open(cluster_count_arr_file, "rb") as fn:
        cluster_count_arr = pickle.load(fn)

    with open("/gene_list.txt") as fh:
        gene_list, gene_ids = zip(*[line.strip().split() for line in fh])

    print(len(gene_list), len(gene_ids))

    # TODO save as pickle so don't need so strip every time
    with gzip.open("/atlas_500umis_mt-1pct_cells.txt.gz", "rt") as fh:
        cell_list = [line.strip() for line in fh]

    print(len(cell_list), len(set(cell_list)))

    # Since the comparison triplet-tuple is so small, not worth exporting one
    # pickle per job. Just export one and then have hail take a range to run (Can
    # just submit one hail job per run, but since startup takes time (docker image
    # loading + pickle loading), can be worth it to amortize cost by running a few
    # diffexp runs per job with nested parallelism)
    with open(input_file, "rb") as handle:
        allcompos_out = pickle.load(handle)
        print("finished loading")

    this_comps = list(itertools.islice(allcompos_out, start, endexc))

    # Make node_tree just an index
    def this_comp_function(k, node_tree):
        return this_comps[k]

    start = time.time()

    # Disable gc within dask, then do every few rouns
    # distributed.utils_perf - WARNING - full garbage collections took 13% CPU time recently (threshold: 10%)
    # See how long takes

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
        print("DONE")

    print(f"Time: {time.time() - start}")

    # Can be lingering dask threads which were keeping hail running even after
    # returned. But happily sys.exit forces all children to die
    client.shutdown()
    print("sys exit")
    sys.exit(0)


if __name__ == "__main__":
    main()
