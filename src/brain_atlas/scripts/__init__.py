import logging

import click
from dask.distributed import Client

from brain_atlas.scripts.filter_mt import main as filter_mt_cmd
from brain_atlas.scripts.make_zarr import main as make_zarr_cmd
from brain_atlas.scripts.subcluster import main as subcluster_cmd
from brain_atlas.scripts.subcluster_p import main as subcluster_p_cmd
from brain_atlas.util import create_logger

log = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Turn on debug logging")
@click.option("--log-file", type=click.Path(writable=True))
@click.option("-d", "--dask-client")
def cli(debug: bool = False, log_file: str = None, dask_client: str = None):
    create_logger(debug=debug, log_file=log_file)
    if dask_client is not None:
        client = Client(dask_client)
        log.debug(f"connected to client {client}")


cli.add_command(filter_mt_cmd, "filter_mt")
cli.add_command(make_zarr_cmd, "make_zarr")
cli.add_command(subcluster_cmd, "subcluster")
cli.add_command(subcluster_p_cmd, "subcluster-p")
