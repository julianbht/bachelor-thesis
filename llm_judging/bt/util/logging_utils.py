# logging_utils.py
from __future__ import annotations
import logging, os, datetime
from typing import Tuple

class _InjectRunKey(logging.Filter):
    def __init__(self, run_key: str):
        super().__init__()
        self.run_key = run_key
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_key"):
            record.run_key = self.run_key
        return True

def setup_run_logger(
    run_key: str,
    *,
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    logger_name: str = "bt.run",  # kept for signature compatibility; no longer used for hierarchy
) -> Tuple[logging.LoggerAdapter, str]:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{ts}_run_{run_key}.log")

    # Configure the *root of your package*
    logger = logging.getLogger("bt")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    inject = _InjectRunKey(run_key)

    # Console handler (shows run key)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.addFilter(inject)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | run=%(run_key)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    # File handler (full detail)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.addFilter(inject)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    adapter = logging.LoggerAdapter(logger, extra={})
    adapter.info("Logging initialized. File: %s", log_path)
    return adapter, log_path
