import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import TextIO

_SECRETS: list[str] = []
_BEARER = re.compile(r"(?i)bearer\s+\S+")


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = dict(getattr(record, "fields", {}))
        event = fields.pop("event", record.getMessage())
        ts = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        extras = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"{ts} {record.levelname} {record.name} {event}"
        if extras:
            line = f"{line} {extras}"
        return _redact(line)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = dict(getattr(record, "fields", {}))
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": fields.pop("event", record.getMessage()),
        }
        payload.update(fields)
        return _redact(json.dumps(payload, default=str))


class Logger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, event: str, **fields: object) -> None:
        self._emit(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: object) -> None:
        self._emit(logging.INFO, event, fields)

    def warning(self, event: str, **fields: object) -> None:
        self._emit(logging.WARNING, event, fields)

    def error(self, event: str, **fields: object) -> None:
        self._emit(logging.ERROR, event, fields)

    def _emit(self, level: int, event: str, fields: dict[str, object]) -> None:
        self._logger.log(level, event, extra={"fields": {"event": event, **fields}})


def get_logger(name: str) -> Logger:
    qualified = name if name.startswith("nemotron.") else f"nemotron.{name}"
    return Logger(logging.getLogger(qualified))


def configure_logging(
    level: str,
    *,
    fmt: str = "json",
    stream: TextIO | None = None,
    secrets: list[str] | None = None,
) -> None:
    global _SECRETS
    _SECRETS = [secret for secret in (secrets or []) if secret]
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(_TextFormatter() if fmt == "text" else _JsonFormatter())
    root.addHandler(handler)


def _redact(text: str) -> str:
    for secret in _SECRETS:
        text = text.replace(secret, "<redacted>")
    return _BEARER.sub("Bearer <redacted>", text)
