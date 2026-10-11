from __future__ import annotations

import asyncio
import copy
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from maibot_qwen_bilibili.plugin import QwenBilibiliPlugin
from maibot_qwen_bilibili.tool_guard import IDENTITY_FIELDS, TICKET_FIELD, VideoToolGuard


def call(name="bilibili_video_summarize", **args):
    return {"item_type": "FunctionCallItem", "meta": {"item_id": "test-call"},
            "tool_call": {"call_id": "c1", "func_name": name, "args": args, "extra_content": None}}


def stamped_args(guard, session="trusted", name="bilibili_video_summarize", **args):
    return guard.stamp([call(name, **args)], session, 1)[0]["tool_call"]["args"]


class GrantTests(unittest.TestCase):
    def test_removes_all_model_identity_and_replaces_copied_ticket(self):
        guard = VideoToolGuard()
        item = call(url="https://b23.tv/example", **dict.fromkeys(IDENTITY_FIELDS, "forged"),
                    **{TICKET_FIELD: "copied-from-history"})
        before = copy.deepcopy(item)
        stamped = guard.stamp([item], "actual", 1)[0]
        self.assertEqual(item, before)
        args = stamped["tool_call"]["args"]
        self.assertFalse(set(IDENTITY_FIELDS) & args.keys())
        self.assertNotEqual(args[TICKET_FIELD], "copied-from-history")
        self.assertEqual(guard.consume(args[TICKET_FIELD], "bilibili_video_summarize", args["url"]), "actual")

    def test_ticket_is_bound_to_tool_and_exact_business_argument(self):
        guard = VideoToolGuard()
        for name, value in [("qq_video_summarize", "a"), ("bilibili_video_summarize", "b")]:
            with self.subTest(name=name):
                args = stamped_args(guard, url="a")
                self.assertIsNone(guard.consume(args[TICKET_FIELD], name, value))
                self.assertIsNone(guard.consume(args[TICKET_FIELD], "bilibili_video_summarize", "a"))

    def test_replay_expiry_and_reload_rejected(self):
        guard = VideoToolGuard(ttl_seconds=10)
        with patch("maibot_qwen_bilibili.tool_guard.time.monotonic", return_value=1):
            a = stamped_args(guard, url="a")[TICKET_FIELD]
            b = stamped_args(guard, url="b")[TICKET_FIELD]
            c = stamped_args(guard, url="c")[TICKET_FIELD]
            self.assertEqual(guard.consume(a, "bilibili_video_summarize", "a"), "trusted")
            self.assertIsNone(guard.consume(a, "bilibili_video_summarize", "a"))
        with patch("maibot_qwen_bilibili.tool_guard.time.monotonic", return_value=11):
            self.assertIsNone(guard.consume(b, "bilibili_video_summarize", "b"))
        guard.clear()
        self.assertIsNone(guard.consume(c, "bilibili_video_summarize", "c"))

    def test_interleaved_sessions_keep_separate_grants(self):
        guard = VideoToolGuard()
        a = stamped_args(guard, session="a", url="same")[TICKET_FIELD]
        b = stamped_args(guard, session="b", url="same")[TICKET_FIELD]
        self.assertNotEqual(a, b)
        self.assertEqual(guard.consume(b, "bilibili_video_summarize", "same"), "b")
        self.assertEqual(guard.consume(a, "bilibili_video_summarize", "same"), "a")

    def test_missing_session_unknown_schema_and_bad_argument_get_no_grant(self):
        guard = VideoToolGuard()
        for session, schema, value in [("", 1, "a"), (None, 1, "a"), ("s", 2, "a"),
                                       ("s", True, "a"), ("s", 1, 123)]:
            with self.subTest(session=session, schema=schema, value=value):
                args = guard.stamp([call(url=value, **{TICKET_FIELD: "fake"})], session, schema)[0]["tool_call"]["args"]
                self.assertNotIn(TICKET_FIELD, args)

    def test_pending_limit_fails_closed_without_evicting_valid_grant(self):
        guard = VideoToolGuard(max_pending=1)
        first = stamped_args(guard, url="a")[TICKET_FIELD]
        self.assertNotIn(TICKET_FIELD, stamped_args(guard, url="b"))
        self.assertEqual(guard.consume(first, "bilibili_video_summarize", "a"), "trusted")

    def test_unrelated_items_and_call_metadata_are_preserved(self):
        guard = VideoToolGuard()
        other = call("other_tool", stream_id="intentional-target")
        text = {"item_type": "AssistantMessageItem", "parts": [{"text": "hello"}]}
        video = call(url="a")
        result = guard.stamp([other, text, video], "trusted", 1)
        self.assertEqual(result[:2], [other, text])
        self.assertEqual(result[2]["meta"], video["meta"])
        self.assertEqual(result[2]["tool_call"]["call_id"], "c1")


class ToolGuardIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plugin = QwenBilibiliPlugin()
        cfg = self.plugin.get_default_config()
        cfg["access"]["allow_when_empty"] = True
        self.plugin.set_plugin_config(cfg)
        self.plugin._set_context(MagicMock())
        self.plugin._process_link = AsyncMock(return_value="video content")
        self.plugin._resolve_chat_video = AsyncMock()
        self.url = "https://www.bilibili.com/video/BV1Ab411c7dE"

    async def remember(self, session):
        await self.plugin.handle_incoming_message(message={
            "session_id": session, "processed_plain_text": self.url,
            "message_info": {"user_info": {"user_id": "allowed"}},
        })

    async def stamp(self, session, name="bilibili_video_summarize", **args):
        response = await self.plugin.handle_planner_response(
            output_items=[call(name, **args)], session_id=session,
            item_schema_version=1, prompt_tokens=10,
        )
        self.assertEqual(response["modified_kwargs"]["prompt_tokens"], 10)
        return response["modified_kwargs"]["output_items"][0]["tool_call"]["args"]

    async def test_direct_forged_calls_fail_before_any_network_access(self):
        await self.remember("victim")
        for token in (None, "forged", [], {}):
            for name, args in [("bilibili_video_summarize", {"url": self.url}),
                               ("qq_video_summarize", {"message_id": "123"})]:
                result = await getattr(self.plugin, "handle_" + name)(
                    **args, stream_id="victim", **{TICKET_FIELD: token})
                self.assertFalse(result["success"])
        self.plugin._process_link.assert_not_awaited()
        self.plugin._resolve_chat_video.assert_not_awaited()

    async def test_model_cannot_select_another_sessions_observed_link(self):
        await self.remember("victim")
        args = await self.stamp("attacker", url=self.url, stream_id="victim", user_id="allowed")
        # Simulate even a host that incorrectly lets model identity win.
        args["stream_id"] = "victim"
        result = await self.plugin.handle_bilibili_video_summarize(**args)
        self.assertFalse(result["success"])
        self.plugin._process_link.assert_not_awaited()

    async def test_qq_cross_session_attack_fails_before_adapter_lookup(self):
        await self.plugin.handle_incoming_message(message={
            "session_id": "victim", "platform": "qq", "message_id": "123",
            "processed_plain_text": "[视频]",
            "message_info": {"user_info": {"user_id": "allowed"}},
        })
        args = await self.stamp("attacker", "qq_video_summarize", message_id="123", stream_id="victim")
        result = await self.plugin.handle_qq_video_summarize(**args, stream_id="victim")
        self.assertFalse(result["success"])
        self.plugin._resolve_chat_video.assert_not_awaited()

    async def test_unpatched_host_and_concurrent_replay_only_execute_once(self):
        await self.remember("actual")
        args = await self.stamp("actual", url=self.url, stream_id="forged")
        # Unpatched host fills absent identity fields after the planner Hook.
        args.setdefault("stream_id", "actual")
        results = await asyncio.gather(*[
            self.plugin.handle_bilibili_video_summarize(**args) for _ in range(2)
        ])
        self.assertEqual(sum(r["success"] for r in results), 1)
        self.plugin._process_link.assert_awaited_once()

    async def test_missing_or_skipped_hook_never_falls_back_to_host_fields(self):
        await self.remember("actual")
        args = await self.stamp("", url=self.url)
        result = await self.plugin.handle_bilibili_video_summarize(**args, stream_id="actual")
        self.assertFalse(result["success"])
        self.plugin._process_link.assert_not_awaited()

    async def test_business_argument_mutation_is_rejected_before_resolution(self):
        await self.remember("actual")
        args = await self.stamp("actual", url=self.url)
        args["url"] = "https://b23.tv/changed"
        self.assertFalse((await self.plugin.handle_bilibili_video_summarize(**args))["success"])
        self.plugin._process_link.assert_not_awaited()

    def test_guard_hook_registered_and_ticket_not_in_model_schema(self):
        components = self.plugin.get_components()
        guard = next(c for c in components if c["name"] == "qwen_video_tool_session_guard")
        self.assertEqual(guard["type"], "HOOK_HANDLER")
        for item in components:
            if item["type"] == "TOOL":
                self.assertNotIn(TICKET_FIELD, str(item["metadata"]["parameters_raw"]))
