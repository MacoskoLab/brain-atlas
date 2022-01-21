import logging
from collections import Counter
from pathlib import Path
from typing import Sequence

import click
import dask
import dask.array as da
import dask.dataframe as dd
import google.auth
import numpy as np

from brain_atlas.util.dataset import Dataset

log = logging.getLogger(__name__)


@click.command("query", no_args_is_help=True)
@click.argument("query", type=int, nargs=-1)
@click.option("--data-path", help="Path to count array", required=True)
@click.option("--cluster-path", help="Path to cluster array", required=True)
@click.option("--metadata-path", help="Path to cell metadata", required=True)
@click.option(
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory for output",
)
@click.option(
    "--subsample",
    type=int,
    default=5000,
    help="Subsample the data per cluster",
    show_default=True,
)
@click.option(
    "--max-cells",
    type=int,
    default=50000,
    help="Max number of cells to return",
    show_default=True,
)
def main(
    query: Sequence[int],
    data_path: str,
    cluster_path: str,
    metadata_path: str,
    output_dir: Path = None,
    subsample: int = 5000,
    max_cells: int = 50000,
):
    """
    Extracts QUERY from a cluster array, subsampling the clusters if needed,
    and creates an npy file containing the counts along with a csv of metadata.

    Can be run on a local zarr directory, or download data from GCS
    """

    credentials, project = google.auth.default()

    log.info(f"Opening count array at {data_path}/counts")
    count_array = da.from_zarr(
        data_path, "counts", storage_options={"token": credentials}
    )
    log.debug(f"Found count array with shape {count_array.shape}")

    log.info(f"Opening cluster labels from {cluster_path}")
    key_array = da.from_zarr(cluster_path, storage_options={"token": credentials})
    log.debug(f"Found label array with shape {key_array.shape}")

    log.info(f"Opening cell metadata from {metadata_path}")
    metadata_df = dd.read_parquet(metadata_path, storage_options={"token": credentials})
    log.debug(f"Found metadata with {metadata_df.shape[1]} columns")

    if len(query):
        query_name = "-".join(map(str, query))
        query = np.array(query)[None, :]

        log.debug(f"Looking for query: {query_name}")
        array_ix = da.compute(np.all(key_array[:, : query.shape[1]] == query, 1))
        if array_ix.sum() == 0:
            log.error("No cells found, quitting")
            return

        log.info(f"Retrieving labels for {array_ix.sum()} cells")
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            key_array = key_array[array_ix, :].compute()
    else:
        log.warning("No query specified, sampling from entire array")
        query_name = "full"
        array_ix = np.ones(key_array.shape[0], dtype=bool)
        key_array = key_array.compute()

    a = key_array > -1
    a_sum = a.sum(1)
    a_argmin = a.argmin(1)
    assert np.all(a.all(1) | (a_sum == a_argmin))

    key_array = [k[:i] for k, i in zip(map(tuple, key_array), a_sum)]
    key_counts = Counter(key_array)
    leaf_keys = sorted(k for k in key_counts if k + (0,) not in key_counts)
    k2i = {k: i for i, k in enumerate(leaf_keys)}

    n_cells = sum(min(key_counts[k], subsample) for k in leaf_keys)
    log.info(f"Found {len(key_array)} cells, sampling {n_cells}")
    if n_cells > max_cells:
        log.error(f"Exceeded limit of {max_cells}, quitting instead!")
        return

    i_ix = np.array([k2i.get(k, -1) for k in key_array])

    log.debug("Generating subsamples")
    subsample_ix = np.zeros(len(key_array), dtype=bool)
    for k in leaf_keys:
        if key_counts[k] < subsample:
            subsample_ix[i_ix == k2i[k]] = True
        else:
            s_ix = np.random.choice(
                (i_ix == k2i[k]).nonzero()[0], size=subsample, replace=False
            )
            subsample_ix[s_ix] = True

    assert subsample_ix.sum() == n_cells
    array_ix[array_ix] = subsample_ix

    log.info(f"Retrieving {n_cells} cells from {data_path}")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        log.debug("downloading array")
        data_array = count_array[array_ix, :].compute()
        log.debug("downloading metadata")
        metadata_df = metadata_df.loc[
            da.array(array_ix).rechunk(Dataset.CHUNKS), :
        ].compute()

    log.info(f"Saving data to {output_dir}")
    np.savez_compressed(
        output_dir / f"subsampled_data_{query_name}.npz", data_array=data_array
    )

    log.info(f"Saving metadata to {output_dir}")
    metadata_df["cluster"] = [
        "-".join(map(str, k)) for k, b in zip(key_array, subsample_ix) if b
    ]
    metadata_df.to_csv(output_dir / f"metadata_{query_name}.csv", index=False)

    log.info("Done!")
