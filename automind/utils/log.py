import sys
import logging


class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)


def setup_logger(cfg):
    """
    Setup logging system with file and console handlers
    Args:
        cfg: Configuration object containing log settings
    Returns:
        logger: Configured logging instance
    """
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )

    # Configure httpx logger
    httpx_logger: logging.Logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Setup main logger
    logger = logging.getLogger("automind")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file handler for normal logs
    file_handler = logging.FileHandler(cfg.log_dir / "automind.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    # Setup file handler for verbose logs
    verbose_file_handler = logging.FileHandler(cfg.log_dir / "automind.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    # Add all handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    return logger
