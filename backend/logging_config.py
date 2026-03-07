"""
TRACE Logging Configuration
---------------------------
Centralized logging setup with enhanced format including
filename, method name, and line number for debugging.
"""

import os
import logging
import sys
from typing import Optional

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

TRACE_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s "
    "%(filename)s:%(funcName)s:%(lineno)d - %(message)s"
)
TRACE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logger with TRACE format."""
    lvl = (level or LOG_LEVEL).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format=TRACE_FORMAT,
        datefmt=TRACE_DATE_FORMAT,
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with TRACE naming."""
    return logging.getLogger(name)


class DecisionLogger:
    """Helper class for logging decisions with context."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_stage(self, stage: int, stage_name: str, **kwargs) -> None:
        """Log stage entry with parameters."""
        params = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"[STAGE {stage}] {stage_name} | {params}")

    def log_decision(self, decision_type: str, result: dict, **context) -> None:
        """Log a decision with result values."""
        self.logger.info(
            f"Decision: {decision_type} | Result: {result} | Context: {context}"
        )

    def log_input(self, func_name: str, **inputs) -> None:
        """Log function inputs."""
        self.logger.debug(f"INPUT {func_name} | {inputs}")

    def log_output(self, func_name: str, **outputs) -> None:
        """Log function outputs."""
        self.logger.debug(f"OUTPUT {func_name} | {outputs}")
