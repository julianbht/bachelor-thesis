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
    logger_name: str = "bt.run",
) -> Tuple[logging.LoggerAdapter, str]:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"run_{run_key}_{ts}.log")

    logger = logging.getLogger(f"{logger_name}.{run_key}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    inject = _InjectRunKey(run_key)

    # Console: include run key
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.addFilter(inject)
    ch.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | run=%(run_key)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(ch)

    # File: no run key, no logger name
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.addFilter(inject)  # harmless; formatter doesn't use it
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    adapter = logging.LoggerAdapter(logger, extra={})
    adapter.info("Logging initialized. File: %s", log_path)
    return adapter, log_path
