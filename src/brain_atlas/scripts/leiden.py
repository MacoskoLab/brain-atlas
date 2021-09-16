import logging
from collections import Counter
from pathlib import Path

import click
import dask.array as da
import igraph as ig
import leidenalg as la
import numpy as np
from dask.distributed import Client

log = logging.getLogger(__name__)


def load_graph(n_cells: int, input_zarr: Path):
    return ig.Graph(
        n=n_cells,
        edges=da.from_zarr(input_zarr, "edges").compute(),
        edge_attrs={"weight": da.from_zarr(input_zarr, "weights").compute()},
    )


def leiden_sweep(graph: ig.Graph, res_list: list[float], cutoff: float = None):
    membership = None
    opt = la.Optimiser()
    membership_arrays = {}
    membership_counts = {}

    for res in res_list:
        log.info(f"Leiden clustering at resolution: {res}")
        partition = la.CPMVertexPartition(
            graph,
            initial_membership=membership,
            weights="weight",
            resolution_parameter=res,
        )
        opt.optimise_partition(partition)
        membership = partition.membership

        membership_arrays[res] = np.array(membership)
        membership_counts[res] = Counter(membership_arrays[res])
        c0c1_ratio = membership_counts[res][0] / membership_counts[res][1]
        if cutoff is not None and c0c1_ratio < cutoff:
            log.info(
                f"Reached nontrivial clustering with c0/c1 ratio {c0c1_ratio:.1f}, stopping"
            )
            break
    else:
        if cutoff is not None:
            log.info(
                f"Finished resolution list without reaching c0/c1 ratio of {cutoff}"
            )

    return membership_arrays, membership_counts


@click.command("leiden-sweep")
@click.argument(
    "graph-zarr",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Path to zarr holding graph edges and weights",
)
@click.option("-n", "--n-cells", required=True, type=int)
@click.option(
    "-o", "--output-dir", required=True, type=click.Path(dir_okay=True, file_okay=False)
)
@click.option("-d", "--dask-client", required=True)
@click.option("--min-res", type=int, default=-9, help="minimum resolution 10^MIN_RES")
@click.option(
    "--max-res", type=int, default=-5, help="maximum resolution 5 x 10^MAX_RES"
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="cluster0/cluster1 ratio cutoff to stop clustering",
)
def main(graph_zarr, n_cells, output_path, dask_client, min_res=-9, max_res=-5):
    client = Client(dask_client)
    log.debug(f"connected to client {client}")

    graph = load_graph(n_cells, graph_zarr)

    res_list = [
        float(f"{b}e{p}") for p in range(min_res, max_res + 1) for b in (1, 2, 5)
    ]

    output_path = Path(output_path)
    m_arrays = leiden_sweep(graph, res_list)
    for res in m_arrays:
        np.save(str(output_path / f"leiden_{res}.npy"), m_arrays[res])
