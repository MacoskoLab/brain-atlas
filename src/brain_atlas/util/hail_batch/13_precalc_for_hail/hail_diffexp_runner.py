import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

import dask
import dask.array as da
import dask.distributed

# Maybe faster, equivalent API
import _pickle as cPickle

import brain_atlas.find_genes as find_genes
import brain_atlas.util.tree as utree
import brain_atlas.util.gcp as gcp

import click

import sys

# FILES NEED
# - atlas_500umis_mt-1pct_cells.txt.gz
# - gene_list.txt
# - 01_mtx_zarr/counts
# - 01_mtx_zarr/atlas_500umis_mt-1pct_merged_clusters.zarr
# - 01_mtx_zarr/atlas_500umis_mt-1pct_cells.txt.gz
# gs://macosko_data/jlanglie/hail_MBASS_files/


def testNumbaParallel():
    import numba
    print(f"Number of numba threads: {numba.get_num_threads()}")
    # python3 -m threadpoolctl -i numpy scipy.linalg
    # -m pdb
    #   b doDiffexp

# If running on local docker for debugging, helpful for copying
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/atlas_500umis_mt-1pct_cells.txt.gz"  25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/gene_list.txt"                       25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/clusters.pickle"             25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/cluster_nz_arr.pickle"       25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/cluster_count_arr.pickle"    25ae21375756:/
# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/hail_diffexp_runner.py"      25ae21375756:/

# sudo  docker  cp  -L  "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/tmpPickleOut.pickle"         25ae21375756:/

@click.command()
@click.option('--inpickle',     required=True,  help='Location of input pickle')
@click.option('--outpickle',    required=True,  help='Location of output pickle')
@click.option('--daskmemlimit', default="12GB", help='Dask mem limit')
@click.option('--daskworkers',  required=True,  help='Number of dask workers to run',              type=int)
@click.option('--daskthreads',  required=True,  help='Number of dask threads p/ worker to run',    type=int)
@click.option('--start',        required=True,  help='Start index in comparison triple',           type=int)
@click.option('--endexc',       required=True,  help='End index (exclusive) in comparison triple', type=int)
def doDiffexp(inpickle, outpickle, daskmemlimit, daskworkers, daskthreads, start, endexc):
    # Debugging how much numba over parallelizes
    import numba
    print(f"Number of numba threads: {numba.get_num_threads()}")
    print(inpickle)
    print(outpickle)

    # array_path = "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/counts"
    # cluster_path = "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/01_mtx_zarr/atlas_500umis_mt-1pct_merged_clusters.zarr"
    array_path   = "gs://macosko_data/jlanglie/hail_MBASS_files/01_mtx_zarr/counts"
    cluster_path = "gs://macosko_data/jlanglie/hail_MBASS_files/01_mtx_zarr/atlas_500umis_mt-1pct_merged_clusters.zarr"
    
    cluster = dask.distributed.LocalCluster(n_workers=daskworkers,
                                            threads_per_worker=daskthreads,
                                            # If increase hail mem size, increase here
                                            memory_limit=daskmemlimit)
    client = dask.distributed.Client(cluster)

    cluster_array = da.from_zarr(cluster_path)
    cluster_array

    count_array = da.from_zarr(array_path)
    count_array

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
      CLUSTER_GENERATION=10
      # As an integer, do CLUSTER_GENERATION*10 + FIRST ELEMENT
      # As a string, just add on str(CLUSTER_GENERATION)+str(FIRST ELEMENT)
      cluster_array[:,0] += CLUSTER_GENERATION*10



    # All these files are precalculated and loaded into root by hail, see `coach` file
    # clustersFN          = "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/clusters.pickle"
    # cluster_nz_arrFN    = "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/cluster_nz_arr.pickle"
    # cluster_count_arrFN = "/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/13_precalc_for_hail/cluster_count_arr.pickle"
    clustersFN = "/clusters.pickle"
    cluster_nz_arrFN = "/cluster_nz_arr.pickle"
    cluster_count_arrFN = "/cluster_count_arr.pickle"

    with open(clustersFN, "rb") as fn:
        clusters = cPickle.load(fn)

    with open(cluster_nz_arrFN, "rb") as fn:
        cluster_nz_arr = cPickle.load(fn)

    with open(cluster_count_arrFN, "rb") as fn:
        cluster_count_arr = cPickle.load(fn)

    with open("/gene_list.txt") as fh:
        gene_list, gene_ids = zip(*[line.strip().split() for line in fh])

    len(gene_list), len(gene_ids)

    import gzip
    # TODO save as pickle so don't need so strip every time
    with gzip.open("/atlas_500umis_mt-1pct_cells.txt.gz", "rt") as fh:
        cell_list = [line.strip() for line in fh]

    print(len(cell_list), len(set(cell_list)))


    # Since the comparison triplet-tuple is so small, not worth exporting one
    # pickle per job. Just export one and then have hail take a range to run (Can
    # just submit one hail job per run, but since startup takes time (docker image
    # loading + pickle loading), can be worth it to amortize cost by running a few
    # diffexp runs per job with nested parallelism)
    with open(inpickle, 'rb') as handle:
        allcompos_out = cPickle.load(handle)
        print("finished loading")


    this_comps = [allcompos_out[i] for i in range(start,endexc)]
    # Make node_tree just an index
    def this_comp_function(k, node_tree):
        # Useful for seeing where at from dask
        print("@@@@@ GOING GOING GOING")
        return this_comps[k]

    import time

    start = time.time()

    # Disable gc within dask, then do every few rouns
    # distributed.utils_perf - WARNING - full garbage collections took 13% CPU time recently (threshold: 10%)
    # See how long takes

    # Didn't work but wish could have plain text progress bar for dask jobs
    # from tqdm.dask import TqdmCallback
    # cb = TqdmCallback(desc="global")
    # print("registering")
    # cb.register()
    # from dask.diagnostics import ProgressBar
    # ProgressBar().register()

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        de_results = dict()
        find_genes.generic_de(this_comp_function,
                              count_array,
                              clusters,
                              # Fake node_tree, get's the k'th element of the
                              # comparison list from this_comp_function
                              node_tree = range(len(this_comps)),
                              cluster_nz = cluster_nz_arr, 
                              cluster_counts = cluster_count_arr,
                              de_results = de_results,
                              # TODO can add as a click param
                              subsample = 80000
        )



    # Write to output path from hail so takes care of exporting to GCS without enlarging docker
    with open(outpickle, 'wb') as handle:
        cPickle.dump(de_results, handle)
        print("DONE")

    print(f'Time: {time.time() - start}')
    
    # Can be lingering dask threads which were keeping hail running even after
    # returned. But happily sys.exit forces all children to die
    print("os exit")
    sys.exit(0)
    print("done exit")

    
if __name__ == '__main__':
    doDiffexp()
