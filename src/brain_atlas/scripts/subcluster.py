import logging
from pathlib import Path

import click
import dask.array as da
import dask_ml.decomposition
import igraph as ig
import numpy as np
from numcodecs import Blosc
from pynndescent import NNDescent

import brain_atlas.neighbors as neighbors
from brain_atlas.gene_selection import dask_pblock
from brain_atlas.scripts.leiden import leiden_sweep

log = logging.getLogger(__name__)


@click.command("subcluster")
@click.argument("input-zarr", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("input-clustering", type=click.Path())
@click.argument("i", type=int)
@click.option(
    "-o", "--output-dir", required=True, type=click.Path(dir_okay=True, file_okay=False)
)
@click.option("--n-pcs", type=int, default=500)
@click.option(
    "-k",
    "--k-neighbors",
    type=int,
    default=250,
)
@click.option(
    "--knn-graph",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Original kNN graph for initialization",
)
@click.option("--min-res", type=int, default=-9, help="minimum resolution 10^MIN_RES")
@click.option(
    "--max-res", type=int, default=-5, help="maximum resolution 5 x 10^MAX_RES"
)
@click.option(
    "--min-gene-diff",
    type=float,
    default=0.025,
    help="Minimum cutoff for calling differential genes",
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="cluster0/cluster1 ratio cutoff to stop clustering",
)
def main(
    input_zarr: str,
    input_clustering: str,
    i: int,
    output_dir: str,
    n_pcs: int = 500,
    k_neighbors: int = 250,
    knn_graph: str = None,
    min_res: int = -9,
    max_res: int = -5,
    min_gene_diff: float = 0.025,
    cutoff: float = None,
):
    """
    Extracts from INPUT-ZARR for INPUT-CLUSTERING == I and subclusters the data
    using a leiden sweep across the specified resolutions, stopping at the optional cutoff.
    """

    data = da.from_zarr(input_zarr)
    output_path = Path(output_dir)

    clusters = np.load(input_clustering)
    ci = clusters == i
    n_cells = ci.sum()

    # compute poisson and select genes
    d_i = data[ci, :]
    exp_pct_nz, pct, ds_p = dask_pblock(d_i)

    gene_cutoff = max(min_gene_diff, np.percentile(exp_pct_nz - pct, 90))
    selected_genes = (exp_pct_nz - pct > gene_cutoff).nonzero()[0]
    n_genes = selected_genes.shape[0]

    # save output
    log.info(f"selected {n_genes} genes")
    gene_output = output_path / f"c{i}_selected_genes.npy"
    log.debug(f"saving to {gene_output}")
    np.save(str(gene_output), selected_genes)

    # subselect cells
    d_i_mem = d_i[:, selected_genes].rechunk((2000, n_genes)).persist()

    # (optional...?) compute PCA on subset genes
    log.info("Computing PCA")
    ipca = (
        dask_ml.decomposition.incremental_pca.IncrementalPCA(
            n_components=n_pcs, batch_size=40000
        )
        .fit_transform(d_i_mem)
        .compute()
    )

    ipca_zarr = output_path / f"c{i}_{n_pcs}-pca.zarr"
    log.debug(f"Saving PCA to {ipca_zarr}")
    da.array(ipca).rechunk((40000, n_pcs)).to_zarr(
        ipca_zarr, compressor=Blosc(cname="lz4hc", clevel=9, shuffle=Blosc.AUTOSHUFFLE)
    )

    if knn_graph is not None:
        log.debug(f"loading existing kNN graph from {knn_graph}")
        original_kng = da.from_zarr(knn_graph, "kng").compute()
        translated_kng = neighbors.translate_kng(ci, original_kng)
    else:
        translated_kng = None

    # compute kNN on PCA (or on genes w/ cosine???) (save to disk)
    log.info("Computing kNN")
    kng, knd = NNDescent(
        ipca, n_neighbors=k_neighbors, metric="euclidean", init_graph=translated_kng
    ).neighbor_graph

    knn_zarr = output_path / f"c{i}_{n_pcs}-pca_{k_neighbors}-knn.zarr"
    log.debug(f"Saving kNN to {knn_zarr}")
    neighbors.write_knn_to_zarr(kng, knd, knn_zarr)

    # compute jaccard on kNN (save to disk)
    log.info("Computing SNN")
    dists = neighbors.compute_jaccard_edges(kng)

    snn_zarr = output_path / f"c{i}_{n_pcs}-pca_{k_neighbors}-snn.zarr"
    log.debug(f"Saving SNN to {snn_zarr}")
    neighbors.write_jaccard_to_zarr(dists, snn_zarr)

    # create igraph from jaccard edges
    log.info("Building graph")
    graph = ig.Graph(
        n=n_cells,
        edges=dists[:, :2],
        edge_attrs={"weight": dists[:, 2]},
    )

    # leiden on igraph on range of resolution values
    # find lowest non-trivial resolution (count0 / count1 < some_max_value)
    res_list = [
        float(f"{b}e{p}") for p in range(min_res, max_res + 1) for b in (1, 2, 5)
    ]

    res_arrays, _ = leiden_sweep(graph, res_list, cutoff)

    clusterings = {}
    for res in sorted(res_arrays):
        # set all cells in other clusters to -1
        all_cells = -1 * np.ones_like(clusters)
        # add cluster assignments for this resolution
        all_cells[ci] = res_arrays[res]
        clusterings[f"{res}"] = all_cells

    clustering_npz = output_path / f"c{i}_{n_pcs}-pca_{k_neighbors}-snn_clusters.npz"
    log.debug(f"Saving clustering output to {clustering_npz}")
    np.savez_compressed(clustering_npz, **clusterings)

    log.info("Done!")
