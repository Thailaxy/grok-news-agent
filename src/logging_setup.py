import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure(level: str | int | None = None) -> None:
    """Configure root logging. Idempotent — safe to call more than once."""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()
    root = logging.getLogger()
    if root.handlers:
        # Already configured — just adjust level.
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=_FORMAT)


class ContextAdapter(logging.LoggerAdapter):
    """Adapter that prepends workflow context (user_id, topic) to each record."""

    def process(self, msg, kwargs):
        ctx = " ".join(f"{k}={v}" for k, v in self.extra.items() if v is not None)
        return (f"[{ctx}] {msg}" if ctx else msg), kwargs


def context_logger(name: str, **extra) -> ContextAdapter:
    return ContextAdapter(logging.getLogger(name), extra)
