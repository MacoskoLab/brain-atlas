import gzip
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def optional_gzip(output_file: str, mode: str = "r"):
    if output_file.endswith(".gz"):
        return gzip.open(output_file, mode + "t")
    else:
        return open(output_file, mode)


def create_logger(debug: bool = False, log_file: Path = None):
    root_log = logging.getLogger()

    # don't need debug output for these
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("gcsfs").setLevel(logging.INFO)
    logging.getLogger("fsspec").setLevel(logging.INFO)
    # google is extra noisy, set to warnings only
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    # matplotlib has a lot of debug output we don't need
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # numba logging is off the charts
    logging.getLogger("numba").setLevel(logging.WARNING)

    if debug:
        root_log.setLevel(logging.DEBUG)
    else:
        root_log.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)

    root_log.addHandler(stream_handler)
    log.debug(msg="Added stream handler")

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_log.addHandler(file_handler)
        log.debug(msg="Added file handler")
