import gzip
import logging

log = logging.getLogger(__name__)


def optional_gzip(output_file: str, mode: str = "r"):
    if output_file.endswith(".gz"):
        return gzip.open(output_file, mode + "t")
    else:
        return open(output_file, mode)


def create_logger(debug: bool = False):
    root_log = logging.getLogger()

    # google is noisy, turn up its logging level
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    # matplotlib has a lot of debug output we don't need
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    if debug:
        root_log.setLevel(logging.DEBUG)
    else:
        root_log.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(formatter)

    root_log.addHandler(stream_handler)
    log.debug(msg="Added stream handler")
