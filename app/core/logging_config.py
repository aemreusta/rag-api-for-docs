"""
Structured logging configuration using structlog.

This module sets up JSON-formatted logging with correlation IDs,
proper masking of sensitive data, and integration with FastAPI middleware.
"""

import gzip
import logging
import logging.handlers
import os
import re
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

try:  # Graceful fallback when structlog is not installed (e.g., minimal test envs)
    import structlog  # type: ignore
    from structlog.typing import EventDict, Processor  # type: ignore

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover - fallback for limited environments
    structlog = None  # type: ignore

    class EventDict(dict):  # type: ignore
        pass

    class Processor:  # type: ignore
        pass

    _HAS_STRUCTLOG = False

from app.core.config import settings

# Context variables for request correlation
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


class CompressingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Custom rotating file handler that compresses rotated log files.
    """

    def doRollover(self) -> None:
        """
        Do a rollover and compress the rotated file.
        """
        super().doRollover()

        # Compress the rotated file (e.g., app.log.1 -> app.log.1.gz)
        if hasattr(self, "rotator") and self.rotator:
            # If a custom rotator is set, use it
            return

        # Default compression behavior
        for i in range(self.backupCount, 0, -1):
            old_log = f"{self.baseFilename}.{i}"
            if os.path.exists(old_log) and not old_log.endswith(".gz"):
                try:
                    with open(old_log, "rb") as f_in:
                        with gzip.open(f"{old_log}.gz", "wb") as f_out:
                            f_out.writelines(f_in)
                    os.remove(old_log)
                except Exception as e:
                    # Log the error but don't crash the application
                    print(f"Failed to compress log file {old_log}: {e}", file=sys.stderr)


class SensitiveDataFilter(logging.Filter):
    """
    Filter that redacts sensitive data patterns from log records.
    """

    # Patterns to redact
    SENSITIVE_PATTERNS = [
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE), "Bearer [REDACTED]"),
        (
            re.compile(r"api[_-]?key\s*[:=]\s*([A-Za-z0-9\-._~+/]{6,})", re.IGNORECASE),
            "api_key=[REDACTED]",
        ),
        (
            re.compile(r"X-API-Key\s*[:=]\s*([A-Za-z0-9\-.]{6,})", re.IGNORECASE),
            "X-API-Key=[REDACTED]",
        ),
        (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL_REDACTED]"),
        (
            re.compile(r'password["\']?\s*[:=]\s*["\']?([^"\'\s]{6,})["\']?', re.IGNORECASE),
            "password=[REDACTED]",
        ),
        (
            re.compile(r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]{6,})["\']?', re.IGNORECASE),
            "secret=[REDACTED]",
        ),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter the log record to redact sensitive data.
        """
        # Apply redaction to the message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)

        # Apply redaction to args if they exist and are a sequence of strings.
        # Important: Do NOT modify mapping args (dict). Some libraries (e.g. Celery)
        # pass a mapping used for %-style formatting. Converting it breaks logging.
        if hasattr(record, "args") and record.args:
            if isinstance(record.args, dict):
                # Leave mapping intact to preserve formatting expectations
                pass
            else:
                try:
                    safe_args = []
                    for arg in record.args:
                        if isinstance(arg, str):
                            safe_arg = arg
                            for pattern, replacement in self.SENSITIVE_PATTERNS:
                                safe_arg = pattern.sub(replacement, safe_arg)
                            safe_args.append(safe_arg)
                        else:
                            safe_args.append(arg)
                    record.args = tuple(safe_args)
                except TypeError:
                    # Non-iterable args, leave as-is
                    pass

        return True


def add_correlation_ids(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add correlation IDs to log events.
    """
    request_id = request_id_var.get("")
    trace_id = trace_id_var.get("")

    if request_id:
        event_dict["request_id"] = request_id
    if trace_id:
        event_dict["trace_id"] = trace_id

    return event_dict


def add_process_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add process information to log events.
    """
    event_dict["process_id"] = os.getpid()
    return event_dict


def mask_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Mask sensitive data in structured log events.
    """
    # Apply masking to the main event message
    if "event" in event_dict and isinstance(event_dict["event"], str):
        for pattern, replacement in SensitiveDataFilter.SENSITIVE_PATTERNS:
            event_dict["event"] = pattern.sub(replacement, event_dict["event"])

    # Apply masking to other string fields
    for key, value in event_dict.items():
        if isinstance(value, str) and key not in [
            "timestamp",
            "level",
            "logger",
            "request_id",
            "trace_id",
        ]:
            for pattern, replacement in SensitiveDataFilter.SENSITIVE_PATTERNS:
                event_dict[key] = pattern.sub(replacement, value)

    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging with JSON format.
    """
    # Get configuration from environment with defensive checks
    log_level = getattr(settings, "LOG_LEVEL", "INFO")
    if hasattr(log_level, "upper"):  # Check if it's a string-like object
        log_level = log_level.upper()
    else:
        log_level = str(log_level).upper()

    log_level_sql = getattr(settings, "LOG_LEVEL_SQL", "WARNING")
    if hasattr(log_level_sql, "upper"):
        log_level_sql = log_level_sql.upper()
    else:
        log_level_sql = str(log_level_sql).upper()

    log_json = getattr(settings, "LOG_JSON", True)
    log_to_file = getattr(settings, "LOG_TO_FILE", False)
    log_file_path = getattr(settings, "LOG_FILE", None)

    # Configure processors based on output format
    processors: list[Processor] = []
    if _HAS_STRUCTLOG:
        processors = [
            structlog.contextvars.merge_contextvars,
            add_correlation_ids,
            add_process_info,
            mask_sensitive_data,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]

    if _HAS_STRUCTLOG:
        if log_json:
            processors.extend(
                [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]
            )
        else:
            processors.extend(
                [
                    structlog.processors.ExceptionPrettyPrinter(),
                    structlog.dev.ConsoleRenderer(colors=True),
                ]
            )

    # Configure structlog
    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

    # Configure standard library logging
    handlers = []

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(SensitiveDataFilter())
    handlers.append(console_handler)

    # File handler (optional)
    if log_to_file and log_file_path:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use pattern with process ID if specified in config
        if "%" in str(log_path):
            # Handle patterns like logs/app_%(process)d_%(asctime)s.log
            import time

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file_resolved = str(log_path).replace("%(process)d", str(os.getpid()))
            log_file_resolved = log_file_resolved.replace("%(asctime)s", timestamp)
        else:
            log_file_resolved = str(log_path)

        file_handler = CompressingRotatingFileHandler(
            filename=log_file_resolved,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.addFilter(SensitiveDataFilter())
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=handlers,
        format="%(message)s",  # structlog handles formatting
        force=True,  # This forces reconfiguration even if already configured
    )

    # Set specific log levels for different components with safe attribute access
    try:
        sql_level = getattr(logging, log_level_sql, logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(sql_level)
    except (TypeError, AttributeError):
        # Fallback if log_level_sql is not a valid string
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class _StdLoggerShim:
    """Thin shim to emulate structlog-style logger interface on stdlib logger.

    Accepts arbitrary keyword arguments and formats them into the message string.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _fmt(self, msg: str, **kwargs) -> str:
        if not kwargs:
            return msg
        kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{msg} | {kv}"

    def debug(self, msg: str, **kwargs) -> None:
        self._logger.debug(self._fmt(msg, **kwargs))

    def info(self, msg: str, **kwargs) -> None:
        self._logger.info(self._fmt(msg, **kwargs))

    def warning(self, msg: str, **kwargs) -> None:
        self._logger.warning(self._fmt(msg, **kwargs))

    def error(self, msg: str, **kwargs) -> None:
        self._logger.error(self._fmt(msg, **kwargs))

    def exception(self, msg: str, **kwargs) -> None:
        self._logger.exception(self._fmt(msg, **kwargs))


def get_logger(name: str = None):
    """
    Get a structured logger instance.

    Args:
        name: Logger name, defaults to caller's module name

    Returns:
        Configured structlog logger
    """
    if name is None:
        # Get the caller's module name
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")

    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    # Fallback to stdlib logger with a shim that accepts structured kwargs
    return _StdLoggerShim(logging.getLogger(name))


def set_request_id(request_id: str = None) -> str:
    """
    Set the request ID for the current context.

    Args:
        request_id: Request ID to set, generates UUID if None

    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    request_id_var.set(request_id)
    return request_id


def set_trace_id(trace_id: str) -> None:
    """
    Set the trace ID for the current context.

    Args:
        trace_id: Trace ID to set
    """
    trace_id_var.set(trace_id)


def get_request_id() -> str:
    """
    Get the current request ID.

    Returns:
        Current request ID or empty string if not set
    """
    return request_id_var.get("")


def get_trace_id() -> str:
    """
    Get the current trace ID.

    Returns:
        Current trace ID or empty string if not set
    """
    return trace_id_var.get("")
