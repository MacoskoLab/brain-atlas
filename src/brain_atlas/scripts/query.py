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

log = logging.getLogger(__name__)


@click.command("query")
@click.argument("query", type=int, nargs=-1)
@click.option("--data-path", help="Path to count array", required=True)
@click.option("--cluster-path", help="Path to cluster array", required=True)
@click.option("--metadata-path", help="Path to cell metadata", required=True)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory for output",
)
@click.option(
    "--subsample", type=int, default=5000, help="Subsample the data per cluster"
)
@click.option(
    "--max-cells", type=int, default=50000, help="Max number of cells to return"
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
    Downloads LEVEL of the ROOT_PATH Leiden tree, performing a sweep across
    the specified resolutions, stopping at the optional cutoff.
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
    log.debug(f"Found metadata with shape {metadata_df.shape}")

    if len(query):
        query = np.array(query)[None, :]
        query_name = "-".join(map(str, query))

        array_ix = np.all(key_array[:, : len(query)] == query, 1).compute()
        log.info(f"Retrieving labels for {array_ix.sum()} cells")
        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            key_array = key_array[array_ix, :].compute()
            count_array = count_array[array_ix, :]
            metadata_df = metadata_df.loc[array_ix]
    else:
        log.warning("No query specified, sampling from entire array")
        query_name = "full"
        key_array = key_array.compute()

    a = key_array > -1
    a_sum = a.sum(1)
    a_argmin = a.argmin(1)
    assert np.all(a.all(1) | (a_sum == a_argmin))

    key_array = [k[:i] for k, i in zip(map(tuple, key_array), a_sum)]
    key_counts = Counter(key_array)

    n_cells = sum(min(v, subsample) for v in key_counts.values())
    log.info(f"Found {len(key_array)} cells, sampling {n_cells}")
    if n_cells > max_cells:
        log.error(f"Exceeded limit of {max_cells}, quitting instead!")
        return

    k2i = {k: i for i, k in enumerate(sorted(set(key_array)))}
    i_ix = np.array([k2i[k] for k in key_array])

    log.debug("Generating subsamples")
    subsample_ix = np.zeros(len(key_array), type=bool)
    for k in k2i:
        if key_counts[k] < subsample:
            subsample_ix[i_ix == k2i[k]] = True
        else:
            s_ix = np.random.choice(
                (i_ix == k2i[k]).nonzero()[0], size=subsample, replace=False
            )
            subsample_ix[s_ix] = True

    assert subsample_ix.sum() == n_cells

    log.info(f"Downloading {n_cells} from {data_path}")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        data_array = count_array[subsample_ix, :].compute()
        metadata_df = metadata_df.loc[subsample_ix, :].compute()

    log.info(f"Saving data to {output_dir}")
    np.save(output_dir / f"subsampled_data_{query_name}.npy", data_array)

    log.info(f"Saving metadata to {output_dir}")
    metadata_df["cluster"] = [
        "-".join(map(str, k)) for k, b in zip(key_array, subsample_ix) if b
    ]
    metadata_df.to_csv(output_dir / f"metadata_{query_name}.csv", index=False)

    log.info("Done!")
