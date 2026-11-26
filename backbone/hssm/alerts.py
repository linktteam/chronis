"""WP9 (owner decisions B3 and C1): developer alerts. A structured event plus one log record for the developers of the integration; nothing user-facing, nothing that identifies a person.

An event is a plain dict: `event` (a kind), `schema_version`, `user_id_hash` (the first 16 hex digits of the SHA-256 of the user id; the raw id is never put in an event or a log line) and the
fields the caller gives (numbers, short strings, lists of reasons). `emit` logs it once as one WARNING on the logger `chronis.hssm.alerts` (the message is JSON) and hands it to every registered
handler (a handler that raises is skipped and counted, it never breaks a fit). Handlers are process-wide and are added with `register_handler`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable

SCHEMA_VERSION = 1
logger = logging.getLogger("chronis.hssm.alerts")
_handlers: list[Callable[[dict], None]] = []
handler_failures = 0
FORBIDDEN_KEYS = ("user_id", "email", "name", "device_id", "raw", "data", "matrix", "X")


def user_id_hash(user_id) -> str:
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:16]


def register_handler(fn: Callable[[dict], None]) -> None:
    _handlers.append(fn)


def clear_handlers() -> None:
    _handlers.clear()


def make_event(kind: str, user_id, **fields) -> dict:
    bad = [k for k in fields if k in FORBIDDEN_KEYS]
    if bad:
        raise ValueError(f"alert fields must not carry personal data or raw data: {bad}")
    return dict(event=str(kind), schema_version=SCHEMA_VERSION, user_id_hash=user_id_hash(user_id), **fields)


def emit(kind: str, user_id, **fields) -> dict:
    """Build, log and dispatch one event. Returns the event."""
    global handler_failures
    ev = make_event(kind, user_id, **fields)
    logger.warning("CHRONIS_DEV_ALERT %s", json.dumps(ev, sort_keys=True, default=str))
    for h in list(_handlers):
        try:
            h(ev)
        except Exception:                                                  # a broken handler must not break a fit
            handler_failures += 1
    return ev
