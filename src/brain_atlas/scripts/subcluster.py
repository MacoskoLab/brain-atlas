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
from brain_atlas.leiden_tree import LeidenTree
from brain_atlas.scripts.leiden import leiden_sweep
from brain_atlas.util.dataset import Dataset

log = logging.getLogger(__name__)


@click.command("subcluster")
@click.argument("root_path", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("level", type=int, nargs=-1)
@click.option("-n", "--n-pcs", type=int, help="Number of PCs to compute, if any")
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
    default=0.025,
    help="Minimum cutoff for calling differential genes",
)
@click.option(
    "--cutoff",
    type=float,
    default=5.0,
    help="Stop clustering when cluster0/cluster1 is below this ratio",
)
@click.option("--resolution", type=str, help="Resolution to use from parent clustering")
@click.option("--overwrite", is_flag=True, help="Don't use any cached results")
@click.option("--high-res", is_flag=True, help="Use a more granular resolution sweep")
def main(
    root_path: str,
    level: Sequence[int],
    n_pcs: int = None,
    k_neighbors: int = None,
    transform: str = None,
    min_res: int = -9,
    max_res: int = -1,
    min_gene_diff: float = 0.025,
    cutoff: float = 5.0,
    resolution: str = None,
    overwrite: bool = False,
    high_res: bool = False,
):
    """
    Subclusters LEVEL of the ROOT_PATH Leiden tree, performing a sweep across
    the specified resolutions, stopping at the optional cutoff.
    """

    root = LeidenTree.from_path(Path(root_path))
    ds = Dataset(str(root.data))

    # open the tree one level up (this might be the root)
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
    assert clusters.shape[0] == ds.counts.shape[0], "Clusters do not match input data"

    tree = LeidenTree(
        root.subcluster_path(level),
        data=root.data,
        n_pcs=n_pcs or root.n_pcs,
        k_neighbors=k_neighbors or root.k_neighbors,
        transform=transform or root.transform,
        resolution=resolution,
    )
    log.debug(f"Saving results to {tree}")

    valid_cache = (not overwrite) and tree.is_valid_cache()
    if not valid_cache:
        tree.write_metadata()

    ci = clusters == level[-1]
    n_cells = ci.sum()
    assert n_cells > 0, "No cells to process"

    log.info(f"Processing {n_cells} / {clusters.shape[0]} cells from {tree.data}")
    d_i = ds.counts[ci, :]

    if valid_cache and tree.selected_genes.exists():
        log.info(f"Loading cached gene selection from {tree.selected_genes}")
        selected_genes = np.load(tree.selected_genes)["selected_genes"]
        n_genes = selected_genes.shape[0]
    else:
        exp_pct_nz, pct, ds_p = dask_pblock(d_i, numis=ds.numis[ci, :])

        gene_cutoff = max(min_gene_diff, np.percentile(exp_pct_nz - pct, 90))
        selected_genes = (exp_pct_nz - pct > gene_cutoff).nonzero()[0]
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
        d_i_mem = tree.transform_fn(d_i[:, selected_genes].rechunk((2000, n_genes)))

    if tree.n_pcs is not None:
        # compute PCA on subset genes
        if tree.n_pcs > n_genes:
            log.error(f"Can't compute {tree.n_pcs} PCs for {n_genes} genes")
            return

        if valid_cache and tree.pca.exists():
            log.info(f"Loading cached PCA from {tree.pca}")
            ipca = da.from_zarr(str(tree.pca)).compute()
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
                str(tree.pca),
                compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE),
                overwrite=overwrite,
            )

        knn_data = ipca
        knn_metric = "euclidean"
    else:
        # NNDescent will convert this to a numpy array
        knn_data = d_i_mem
        knn_metric = "cosine"

    if valid_cache and tree.knn.exists():
        log.info(f"Loading cached kNN from {tree.knn}")
        kng = da.from_zarr(str(tree.knn), "kng")
    else:
        translated_kng = None
        if parent.knn.exists():
            log.debug(f"loading existing kNN graph from {parent.knn}")
            original_kng = da.from_zarr(str(parent.knn), "kng")
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

        # compute kNN, either on PCA or on counts
        log.info("Computing kNN")
        kng, knd = NNDescent(
            data=knn_data,
            n_neighbors=tree.k_neighbors + 1,
            metric=knn_metric,
            init_graph=translated_kng,
        ).neighbor_graph

        # remove self-edges
        kng = kng[:, 1:].astype(np.int32)
        knd = knd[:, 1:]

        log.debug(f"Saving kNN to {tree.knn}")
        neighbors.write_knn_to_zarr(kng, knd, tree.knn, overwrite=overwrite)

    if valid_cache and tree.snn.exists():
        log.info(f"Loading cached SNN from {tree.snn}")

        edges = da.from_zarr(str(tree.snn), "edges").compute()
        weights = da.from_zarr(str(tree.snn), "weights").compute()
    else:
        # compute jaccard on kNN
        log.info("Computing SNN")
        dists = neighbors.compute_jaccard_edges(kng)

        log.debug(f"Saving SNN to {tree.snn}")
        neighbors.write_jaccard_to_zarr(dists, tree.snn, overwrite=overwrite)

        edges = dists[:, :2]
        weights = dists[:, 2]

    # create igraph from jaccard edges
    log.info("Building graph")
    graph = ig.Graph(n=n_cells, edges=edges, edge_attrs={"weight": weights})

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
