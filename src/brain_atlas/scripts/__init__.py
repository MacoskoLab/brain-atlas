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
@click.option("--log-file", type=click.Path(writable=True, path_type=Path))
@click.option("-d", "--dask-client")
def cli(debug: bool = False, log_file: Path = None, dask_client: str = None):
    create_logger(debug=debug, log_file=log_file)
    if dask_client is None:
        dask_client = dask.distributed.client._get_global_client()

    if dask_client is not None:
        client = Client(dask_client)
        log.debug(f"connected to client {client.scheduler.address}")
    else:
        log.debug("No dask server detected")


cli.add_command(filter_mt_cmd, "filter-mt")
cli.add_command(gcs_to_zarr_cmd, "gcs-to-zarr")
cli.add_command(make_zarr_cmd, "make-zarr")
cli.add_command(query_cmd, "query")
cli.add_command(subcluster_cmd, "subcluster")
