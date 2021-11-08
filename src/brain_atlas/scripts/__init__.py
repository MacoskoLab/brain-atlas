import logging

import click
from dask.distributed import Client

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
