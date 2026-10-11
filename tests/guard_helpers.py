"""Drive the real planner Hook before invoking a tool in existing regressions."""

from __future__ import annotations

from typing import Any


async def guarded_call(plugin: Any, name: str, *, stream_id: str = "", **kwargs: Any) -> dict:
    business_key = "url" if name == "bilibili_video_summarize" else "message_id"
    result = await plugin.handle_planner_response(
        session_id=stream_id, item_schema_version=1,
        output_items=[{"item_type": "FunctionCallItem", "tool_call": {
            "call_id": "fixture-call", "func_name": name,
            "args": {business_key: kwargs.get(business_key, "")},
        }}],
    )
    args = result.get("modified_kwargs", {}).get("output_items", [{"tool_call": {"args": {}}}])[0]["tool_call"]["args"]
    return await getattr(plugin, "handle_" + name)(**{**kwargs, **args, "stream_id": stream_id})
