"""Bind video calls to the session delivered by the trusted planner Hook.

The host may merge model arguments over its own identity fields. Never use those
fields as an authorization source. Each Hook-stamped call instead consumes one
short-lived, opaque grant from this plugin instance before any network access.
"""

from __future__ import annotations

import secrets
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

TICKET_FIELD = "_qwen_call_ticket"
TOOL_ARGUMENTS = {"bilibili_video_summarize": "url", "qq_video_summarize": "message_id"}
IDENTITY_FIELDS = ("stream_id", "chat_id", "session_id", "user_id", "group_id", "platform")


@dataclass(frozen=True, slots=True)
class _Grant:
    session_id: str
    tool_name: str
    argument: str
    expires_at: float


class VideoToolGuard:
    def __init__(self, *, ttl_seconds: float = 300, max_pending: int = 512) -> None:
        self._ttl = ttl_seconds
        self._limit = max_pending
        self._pending: dict[str, _Grant] = {}

    def clear(self) -> None:
        self._pending.clear()

    def _prune(self) -> None:
        now = time.monotonic()
        for token, grant in list(self._pending.items()):
            if grant.expires_at <= now:
                self._pending.pop(token, None)

    def stamp(self, output_items: Any, session_id: Any, schema_version: Any) -> Any:
        """Accept only the public v1 Context Item Hook payload; never mutate input."""
        if not isinstance(output_items, list):
            return output_items
        self._prune()
        result = deepcopy(output_items)
        supported = type(schema_version) is int and schema_version == 1
        session = session_id.strip() if isinstance(session_id, str) else ""
        for item in result:
            if not isinstance(item, dict) or item.get("item_type") != "FunctionCallItem":
                continue
            call = item.get("tool_call")
            if not isinstance(call, dict):
                continue
            name = call.get("func_name")
            if not isinstance(name, str) or name not in TOOL_ARGUMENTS:
                continue
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            # Strip both forged identities and any ticket copied from history.
            for key in (*IDENTITY_FIELDS, TICKET_FIELD):
                args.pop(key, None)
            key = TOOL_ARGUMENTS[name]
            value = args.get(key, "")
            if not supported or not session or not isinstance(value, str):
                continue
            if len(self._pending) >= self._limit:
                continue  # Fail closed rather than discard a still-pending call.
            token = secrets.token_urlsafe(32)
            self._pending[token] = _Grant(session, name, value, time.monotonic() + self._ttl)
            args[TICKET_FIELD] = token
        return result

    def consume(self, token: Any, tool_name: str, argument: Any) -> str | None:
        """Consume synchronously, before any await; replay and mutations fail closed."""
        self._prune()
        if not isinstance(token, str):
            return None
        grant = self._pending.pop(token, None)
        if grant is None or grant.tool_name != tool_name or grant.argument != argument:
            return None
        return grant.session_id
