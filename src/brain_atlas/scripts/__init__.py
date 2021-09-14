import click

from ..util import create_logger
from .filter_mt import main as filter_mt_cmd
from .make_zarr import main as make_zarr_cmd


@click.group()
@click.option("--debug", is_flag=True, help="Turn on debug logging")
def cli(debug: bool = False):
    create_logger(debug)


cli.add_command(filter_mt_cmd, "filter_mt")
cli.add_command(make_zarr_cmd, "make_zarr")
