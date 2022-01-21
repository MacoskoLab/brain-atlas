import logging
from pathlib import Path

import click
import dask.distributed
from dask.distributed import Client

from brain_atlas.scripts.filter_mt import main as filter_mt_cmd
from brain_atlas.scripts.gcs_to_zarr import main as gcs_to_zarr_cmd
from brain_atlas.scripts.make_zarr import main as make_zarr_cmd
from brain_atlas.scripts.query import main as query_cmd
from brain_atlas.scripts.subcluster import main as subcluster_cmd
from brain_atlas.util import create_logger

log = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Turn on debug logging")
@click.option("-l", "--log-file", type=click.Path(writable=True, path_type=Path))
@click.option("-d", "--dask-client")
@click.option(
    "--no-dask/--dask",
    "start_cluster",
    is_flag=True,
    help="Start a Dask cluster if one can't be found",
    default=False,
)
def cli(
    debug: bool = False,
    log_file: Path = None,
    dask_client: str = None,
    start_cluster: bool = False,
):
    create_logger(debug=debug, log_file=log_file)
    if dask_client is None:
        client = dask.distributed.client._get_global_client()
        if client is None:
            log.info("No Dask cluster found")
            if start_cluster:
                log.info("Starting cluster on local machine")
                client = dask.distributed.Client()
    else:
        client = Client(dask_client)

    if client is not None:
        log.debug(f"connected to client {client.scheduler.address}")


cli.add_command(filter_mt_cmd, "filter-mt")
cli.add_command(gcs_to_zarr_cmd, "gcs-to-zarr")
cli.add_command(make_zarr_cmd, "make-zarr")
cli.add_command(query_cmd, "query")
cli.add_command(subcluster_cmd, "subcluster")
