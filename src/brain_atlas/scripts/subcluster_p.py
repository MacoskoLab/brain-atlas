import logging
from pathlib import Path
from typing import Sequence

import click
import dask
import dask.array as da
import igraph as ig
import numpy as np
from pynndescent import NNDescent

import brain_atlas.neighbors as neighbors
from brain_atlas.gene_selection import dask_pblock
from brain_atlas.leiden import leiden_sweep
from brain_atlas.leiden_tree import LeidenTree
from brain_atlas.util.dataset import Dataset

log = logging.getLogger(__name__)


@click.command("subcluster-p")
@click.argument("root-path", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("level", type=int, nargs=-1)
@click.option("-k", "--k-neighbors", type=int)
@click.option(
    "-t",
    "--transform",
    type=click.Choice(["none", "sqrt", "log1p"], case_sensitive=False),
)
@click.option("--min-res", type=int, default=-9, help="Minimum resolution 10^MIN_RES")
@click.option("--max-res", type=int, default=-1, help="Maximum resolution 5x10^MAX_RES")
@click.option(
    "--min-gene-diff",
    type=float,
    default=0.05,
    help="Minimum cutoff for calling differential genes",
)
@click.option("--resolution", type=str, help="Resolution to use from parent clustering")
@click.option("--overwrite", is_flag=True, help="Don't use any cached results")
@click.option("--high-res", is_flag=True, help="Use a more granular resolution sweep")
def main(
    root_path: str,
    level: Sequence[int],
    k_neighbors: int = None,
    transform: str = None,
    min_res: int = -9,
    max_res: int = -1,
    min_gene_diff: float = 0.05,
    resolution: str = None,
    overwrite: bool = False,
    high_res: bool = False,
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

    tree = LeidenTree(
        root.subcluster_path(level),
        data=root.data,
        n_pcs=None,
        k_neighbors=k_neighbors or root.k_neighbors,
        transform=transform or root.transform,
        scaled=False,
        jaccard=False,
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

    log.info(f"Processing {n_cells} / {clusters.shape[0]} cells from {tree.data}")
    d_i = ds.counts[ci, :]

    if valid_cache and tree.selected_genes.exists():
        log.info(f"Loading cached gene selection from {tree.selected_genes}")
        with np.load(tree.selected_genes) as d:
            selected_genes = d["selected_genes"]
    else:
        exp_pct_nz, pct, ds_p = dask_pblock(d_i, numis=ds.numis[ci, :])

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
        knn_data = tree.transform_fn(d_i[:, selected_genes]).compute()

    if knn_data.shape[0] < tree.k_neighbors ** 2:
        # for small arrays, it is faster to compute the full pairwise distance
        log.info("Computing all-by-all edge list")
        edges = neighbors.cosine_edgelist(knn_data)
    else:
        if valid_cache and tree.knn.exists():
            log.info(f"Loading cached kNN from {tree.knn}")
            kng = da.from_zarr(tree.knn, "kng").compute()
            knd = da.from_zarr(tree.knn, "knd").compute()
        else:
            translated_kng = None
            if parent.knn.exists():
                log.debug(f"loading existing kNN graph from {parent.knn}")
                original_kng = da.from_zarr(parent.knn, "kng")
                if original_kng.shape[0] == ci.shape[0]:
                    translated_kng = neighbors.translate_kng(ci, original_kng.compute())
                elif original_kng.shape[0] == (clusters > -1).sum():
                    translated_kng = neighbors.translate_kng(
                        ci[clusters > -1], original_kng.compute()
                    )
                else:
                    log.warning(
                        f"kNN shape {original_kng.shape} did not match clusters {ci.shape}"
                    )

            # compute kNN
            log.info("Computing kNN")
            kng, knd = NNDescent(
                data=knn_data,
                n_neighbors=tree.k_neighbors + 1,
                metric="cosine",
                init_graph=translated_kng,
            ).neighbor_graph

            kng = kng.astype(np.int32)
            log.debug(f"Saving kNN to {tree.knn}")
            neighbors.write_knn_to_zarr(kng, knd, tree.knn, overwrite=overwrite)

        log.info("Creating edge list")
        edges = neighbors.kng_to_edgelist(kng, knd)

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
    else:
        cached_arrays = None

    # leiden on igraph on range of resolution values
    # find lowest non-trivial resolution (count0 / count1 < some_max_value)
    res_list = [float(f"{b}e{p}") for p in range(min_res, max_res + 1) for b in bs]
    res_arrays, _ = leiden_sweep(
        graph, res_list, np.sqrt(n_cells), cached_arrays=cached_arrays
    )

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