import logging
from collections import Counter

import click
import dask.array as da
import igraph as ig
import leidenalg as la
import numpy as np

log = logging.getLogger(__name__)


def load_graph(n_cells: int, input_zarr: str):
    return ig.Graph(
        n=n_cells,
        edges=da.from_zarr(input_zarr, "edges").compute(),
        edge_attrs={"weight": da.from_zarr(input_zarr, "weights").compute()},
    )


def leiden_sweep(
    graph: ig.Graph,
    res_list: list[float],
    cutoff: float = None,
    cached_arrays: dict[float, np.ndarray] = None,
):
    membership = None
    opt = la.Optimiser()

    membership_arrays = {}
    membership_counts = {}
    if cached_arrays is not None:
        log.debug(f"Got {len(cached_arrays)} cached membership arrays")
        membership_arrays.update(cached_arrays)
        membership_counts.update((k, Counter(v)) for k, v in cached_arrays.items())

    for res in res_list:
        log.info(f"Leiden clustering at resolution: {res}")
        if res in membership_arrays:
            membership = membership_arrays[res]
        else:
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

        if membership_counts[res][1] > 0:
            c0c1_ratio = membership_counts[res][0] / membership_counts[res][1]
        else:
            continue

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
@click.argument("graph-zarr", type=click.Path(dir_okay=True, file_okay=False))
@click.option("-n", "--n-cells", required=True, type=int)
@click.option("-o", "--output-file", required=True, type=click.Path())
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
def main(graph_zarr, n_cells, output_file, min_res=-9, max_res=-5, cutoff=None):
    graph = load_graph(n_cells, graph_zarr)

    res_list = [
        float(f"{b}e{p}") for p in range(min_res, max_res + 1) for b in (1, 2, 5)
    ]

    m_arrays, _ = leiden_sweep(graph, res_list, cutoff=cutoff)

    m_arrays = {f"{res}": arr for res, arr in m_arrays.items()}
    np.savez_compressed(output_file, **m_arrays)
