import logging

import click
from dask.distributed import Client

from ..util import create_logger
from .filter_mt import main as filter_mt_cmd
from .leiden import main as leiden_cmd
from .make_zarr import main as make_zarr_cmd
from .subcluster import main as subcluster_cmd

log = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Turn on debug logging")
@click.option("-d", "--dask-client")
def cli(debug: bool = False, dask_client: str = None):
    create_logger(debug)
    if dask_client is not None:
        client = Client(dask_client)
        log.debug(f"connected to client {client}")


cli.add_command(filter_mt_cmd, "filter_mt")
cli.add_command(make_zarr_cmd, "make_zarr")
cli.add_command(leiden_cmd, "leiden")
cli.add_command(subcluster_cmd, "subcluster")
