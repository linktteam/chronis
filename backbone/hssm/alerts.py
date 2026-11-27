"""WP9 (owner decisions B3 and C1): developer alerts. A structured event plus one log record for the developers of the integration; nothing user-facing, nothing that identifies a person.

An event is a plain dict: `event` (a kind), `schema_version`, `user_id_hash` (the first 16 hex digits of the SHA-256 of the user id; the raw id is never put in an event or a log line) and the
fields the caller gives (numbers, short strings, lists of reasons). `emit` logs it once as one WARNING on the logger `chronis.hssm.alerts` (the message is JSON) and hands it to every registered
handler (a handler that raises is skipped and counted, it never breaks a fit). Handlers are process-wide and are added with `register_handler`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Callable

SCHEMA_VERSION = 1
logger = logging.getLogger("chronis.hssm.alerts")
_handlers: list[Callable[[dict], None]] = []
handler_failures = 0
FORBIDDEN_KEYS = ("user_id", "email", "name", "device_id", "raw", "data", "matrix", "X")
ASSESSMENT_CRASHED = "ASSESSMENT_CRASHED"           # WP10B F3: an assessment of the lineage (K change, fit stability, scaler, matching) crashed or returned a malformed record; the lineage failed closed
DEDUP_MAX = 50_000                                  # distinct (kind, user, key) triples remembered; beyond that the OLDEST is forgotten, so a very old crash may alert again
_seen: "OrderedDict[tuple, None]" = OrderedDict()
_seen_lock = threading.Lock()
_CLASS_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")


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
        except Exception as exc:                                           # a broken handler must not break a fit
            handler_failures += 1
            try:                                                           # WP10B F3: a failure to deliver is logged (class name only: no message text can leak) and never propagates
                logger.warning("CHRONIS_DEV_ALERT handler failed (kind=%s, error=%s)", kind, exception_class_name(exc))
            except Exception:                                              # noqa: BLE001
                pass
    return ev


def exception_class_name(exc_or_name) -> str:
    """The class name of an exception (or a given name), restricted to identifier characters (at most 80). Anything else becomes `Exception`: a name can never carry message text."""
    name = exc_or_name if isinstance(exc_or_name, str) else type(exc_or_name).__name__
    return name if _CLASS_NAME.fullmatch(name or "") else "Exception"


def reset_dedup() -> None:
    with _seen_lock:
        _seen.clear()


def emit_once(kind: str, user_id, dedup_key, **fields) -> dict | None:
    """`emit`, at most once per (kind, user, `dedup_key`) in this process; returns None when the alert was already sent. `dedup_key=None` (the caller does not know enough to deduplicate) always emits."""
    if dedup_key is not None:
        key = (str(kind), user_id_hash(user_id), dedup_key)
        with _seen_lock:
            if key in _seen:
                return None
            _seen[key] = None
            while len(_seen) > DEDUP_MAX:
                _seen.popitem(last=False)
    return emit(kind, user_id, **fields)
