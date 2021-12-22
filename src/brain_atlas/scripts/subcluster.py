import logging
from pathlib import Path
from typing import Sequence

import click
import dask
import dask.array as da
import dask_ml.decomposition
import igraph as ig
import numpy as np
from numcodecs import Blosc
from pynndescent import NNDescent

import brain_atlas.neighbors as neighbors
from brain_atlas.gene_selection import dask_pblock
from brain_atlas.leiden import leiden_sweep
from brain_atlas.leiden_tree import LeidenTree
from brain_atlas.util.dataset import Dataset

log = logging.getLogger(__name__)


@click.command("subcluster")
@click.argument("root-path", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("level", type=int, nargs=-1)
@click.option("-n", "--n-pcs", type=int, help="Number of PCs to compute, if any")
@click.option("-k", "--k-neighbors", type=int)
@click.option(
    "-t",
    "--transform",
    type=click.Choice(["none", "sqrt", "log1p"], case_sensitive=False),
)
@click.option(
    "-z/-Z",
    "--std/--no-std",
    "scaled",
    help="Standardize genes before PCA/kNN",
    default=None,
)
@click.option(
    "--snn/--no-snn",
    "jaccard",
    help="Compute shared nearest neighbors graph",
    default=None,
)
@click.option("--min-res", type=int, default=-9, help="Minimum resolution 10^MIN_RES")
@click.option("--max-res", type=int, default=-1, help="Maximum resolution 5x10^MAX_RES")
@click.option(
    "--min-gene-diff",
    type=float,
    default=0.05,
    help="Minimum cutoff for calling differential genes",
)
@click.option(
    "--cutoff",
    type=float,
    help="Stop clustering when cluster0/cluster1 is below this ratio",
)
@click.option("--resolution", type=str, help="Resolution to use from parent clustering")
@click.option("--overwrite", is_flag=True, help="Don't use any cached results")
@click.option("--high-res", is_flag=True, help="Use a more granular resolution sweep")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Alternate directory for output",
)
@click.option(
    "--max-array-size",
    type=int,
    default=40000,
    help="Threshold for using in-memory/brute-force algorithms",
)
def main(
    root_path: str,
    level: Sequence[int],
    n_pcs: int = None,
    k_neighbors: int = None,
    transform: str = None,
    scaled: bool = None,
    jaccard: bool = None,
    min_res: int = -9,
    max_res: int = -1,
    min_gene_diff: float = 0.05,
    cutoff: float = None,
    resolution: str = None,
    overwrite: bool = False,
    high_res: bool = False,
    output_dir: str = None,
    max_array_size: int = 40000,
):
    """
    Subclusters LEVEL of the ROOT_PATH Leiden tree, performing a sweep across
    the specified resolutions, stopping at the optional cutoff.
    """

    root = LeidenTree.from_path(Path(root_path))
    ds = Dataset(root.data)

    # check if we are clustering the root (e.g. no parent)
    if len(level) == 0:
        parent = root  # for cache purposes

        log.debug("Using all-zero clustering for root")
        clusters = np.zeros(ds.counts.shape[0], dtype=np.int32)
        ci = np.ones_like(clusters, dtype=bool)
    else:
        # open the tree one level up
        parent = LeidenTree.from_path(root.subcluster_path(level[:-1]))
        clusters = np.load(parent.clustering)
        if resolution is None:
            if parent.resolution is None:
                # use the largest resolution present in the file
                resolution = max(clusters, key=float)
            else:
                resolution = parent.resolution

        log.debug(f"Using parent clustering with resolution {resolution}")
        clusters: np.ndarray = clusters[resolution]
        ci = clusters == level[-1]
        assert ci.shape[0] == ds.counts.shape[0], "Clusters do not match input data"

    if output_dir is None:
        output_dir = root.subcluster_path(level)

    tree = LeidenTree(
        output_dir,
        data=root.data,
        n_pcs=n_pcs or root.n_pcs,
        k_neighbors=k_neighbors or root.k_neighbors,
        transform=transform or root.transform,
        scaled=scaled if scaled is not None else root.scaled,
        jaccard=jaccard if jaccard is not None else root.jaccard,
        resolution=None,
    )
    log.debug(f"Saving results to {tree}")

    valid_cache = (not overwrite) and tree.is_valid_cache()
    if not valid_cache:
        tree.write_metadata()
    else:
        log.debug("Using cached values")

    n_cells = ci.sum()
    assert n_cells > 0, "No cells to process"
    if cutoff is None:
        cutoff = np.sqrt(n_cells)

    log.info(f"Processing {n_cells} / {clusters.shape[0]} cells from {tree.data}")
    d_i = ds.counts[ci, :]
    n_i = ds.numis[ci, :]
    if n_cells < max_array_size:
        log.debug("computing in memory")
        d_i, n_i = da.compute(d_i, n_i)

    if valid_cache and tree.selected_genes.exists():
        log.info(f"Loading cached gene selection from {tree.selected_genes}")
        with np.load(tree.selected_genes) as d:
            selected_genes = d["selected_genes"]
        n_genes = selected_genes.shape[0]
    else:
        exp_pct_nz, pct, ds_p = dask_pblock(d_i, numis=n_i)

        selected_genes = ((exp_pct_nz - pct > min_gene_diff) & (ds_p < -5)).nonzero()[0]
        n_genes = selected_genes.shape[0]

        # save output
        log.info(f"Selected {n_genes} genes")
        log.debug(f"saving to {tree.selected_genes}")
        np.savez_compressed(
            tree.selected_genes,
            exp_pct_nz=exp_pct_nz,
            pct=pct,
            ds_p=ds_p,
            selected_genes=selected_genes,
        )

    # subselect genes
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        d_i_mem = tree.transform_data(d_i[:, selected_genes])
        if isinstance(d_i_mem, da.Array):
            d_i_mem = d_i_mem.rechunk((2000, n_genes))

    if scaled:
        d_i_mem = d_i_mem / np.std(d_i_mem, axis=0, keepdims=True)

    if tree.n_pcs is None:
        # NNDescent will convert this to a numpy array
        knn_data = d_i_mem
        knn_metric = "cosine"
    else:
        # compute PCA on subset genes
        if tree.n_pcs > n_genes:
            log.error(f"Can't compute {tree.n_pcs} PCs for {n_genes} genes")
            return

        if valid_cache and tree.pca.exists():
            log.info(f"Loading cached PCA from {tree.pca}")
            ipca = da.from_zarr(tree.pca).compute()
        else:
            log.info("Computing PCA")
            ipca = (
                dask_ml.decomposition.incremental_pca.IncrementalPCA(
                    n_components=tree.n_pcs, batch_size=40000
                )
                .fit_transform(d_i_mem)
                .compute()
            )

            log.debug(f"Saving PCA to {tree.pca}")
            da.array(ipca).rechunk((40000, n_pcs)).to_zarr(
                tree.pca,
                compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
                overwrite=overwrite,
            )

        knn_data = ipca
        knn_metric = "euclidean"

    if n_cells < max_array_size and not tree.n_pcs:
        # for small arrays, it is faster to compute the full pairwise distance
        log.info("Computing edge list via brute-force algo")
        if tree.jaccard:
            log.debug("calculating jaccard scores")
            edges = neighbors.k_jaccard_edgelist(knn_data, k=k_neighbors + 1)
        else:
            edges = neighbors.k_cosine_edgelist(knn_data, k=k_neighbors + 1)
    else:
        if valid_cache and tree.knn.exists():
            log.info(f"Loading cached kNN from {tree.knn}")
            kng = da.from_zarr(tree.knn, "kng").compute()
            knd = None  # will load if needed for edge list
        else:
            # compute kNN, either on PCA or on counts
            log.info("Computing kNN via pynndescent")
            kng, knd = NNDescent(
                data=knn_data,
                n_neighbors=tree.k_neighbors + 1,
                metric=knn_metric,
                pruning_degree_multiplier=2.0,
                diversify_prob=0.0,
            ).neighbor_graph

            kng = kng.astype(np.int32)
            log.debug(f"Saving kNN to {tree.knn}")
            neighbors.write_knn_to_zarr(kng, knd, tree.knn, overwrite=overwrite)

        if tree.jaccard:
            # compute jaccard on kNN
            log.info("Computing SNN")
            edges = neighbors.kng_to_jaccard(kng)
        else:
            if knd is None:
                log.debug(f"Loading kNN distances from {tree.knn}")
                knd = da.from_zarr(tree.knn, "knd").compute()

            log.info("Creating edge list")
            edges = neighbors.kng_to_edgelist(kng, 1 - knd)

    # create igraph from edge list
    log.info(f"Building graph with {edges.shape[0]} edges")
    graph = ig.Graph(n=n_cells, edges=edges[:, :2], edge_attrs={"weight": edges[:, 2]})

    if high_res:
        bs = range(1, 10)
    else:
        bs = (1, 2, 5)

    if valid_cache and tree.clustering.exists():
        with np.load(tree.clustering) as data:
            cached_arrays = {float(k): d[d > -1] for k, d in data.items()}

        if not all(v.shape[0] == n_cells for v in cached_arrays.values()):
            log.warning("cached arrays are the wrong size")
            cached_arrays = None
    else:
        cached_arrays = None

    # leiden on igraph on range of resolution values
    # find lowest non-trivial resolution (count0 / count1 < some_max_value)
    res_list = [float(f"{b}e{p}") for p in range(min_res, max_res + 1) for b in bs]
    res_arrays, _ = leiden_sweep(graph, res_list, cutoff, cached_arrays=cached_arrays)

    clusterings = {}
    for res in sorted(res_arrays):
        # set all cells in other clusters to -1
        all_cells = -1 * np.ones_like(clusters)
        # add cluster assignments for this resolution
        all_cells[ci] = res_arrays[res]
        clusterings[f"{res}"] = all_cells

    log.debug(f"Saving clustering output to {tree.clustering}")
    np.savez_compressed(tree.clustering, **clusterings)

    tree.resolution = max(clusterings, key=float)
    log.info(f"Writing new metadata with resolution {tree.resolution}")
    tree.write_metadata()

    log.info("Done!")
