from __future__ import annotations

from maibot_qwen_bilibili.tests.guard_helpers import guarded_call
import copy
import time
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

from maibot_qwen_bilibili.chat_video import ChatVideoError, resolve_chat_video
from maibot_qwen_bilibili.plugin import QwenBilibiliPlugin
from maibot_qwen_bilibili.qwen_client import QwenResult, QwenTimeoutError


def incoming(message_id: str = "123", *, stream: str = "group-2", user: str = "1", group: str = "2") -> dict:
    return {
        "message_id": message_id,
        "platform": "qq",
        "session_id": stream,
        "processed_plain_text": "[视频] 文件: test.mp4，大小: 1234",
        "message_info": {"user_info": {"user_id": user}, "group_info": {"group_id": group}},
    }


def original(message_id: int = 123, *, user: int = 1, group: int | None = 2) -> dict:
    return {
        "message_id": message_id,
        "message_type": "group" if group is not None else "private",
        "sender": {"user_id": user},
        **({"group_id": group} if group is not None else {}),
        "message": [{"type": "video", "data": {
            "file": "opaque-video-file-id",
            "url": "https://gchat.qpic.cn/video/test.mp4?token=private-signed-token",
            "file_size": 1234,
        }}],
    }


class ChatVideoPluginTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.plugin = QwenBilibiliPlugin()
        self.config = self.plugin.build_default_config()
        self.config["access"]["user_whitelist"] = ["1"]
        self.config["access"]["group_whitelist"] = ["2"]
        self.config["access"]["whitelist_mode"] = "all"
        self.config["qwen"]["api_key"] = "test-only-key"
        self.plugin.set_plugin_config(self.config)
        self.context = MagicMock()
        self.context.api.call = AsyncMock(return_value=original())
        self.plugin._set_context(self.context)
        self.plugin._get_or_create_analysis = AsyncMock(return_value={"text": "测试视频内容"})

    async def remember(self, **kwargs):
        message = incoming(**kwargs)
        await self.plugin.handle_incoming_message(message=message)
        return self.plugin._get_chat_video_authorization(message["session_id"], message["message_id"])

    async def test_hook_is_local_and_preserves_message(self):
        message = incoming()
        before = copy.deepcopy(message)
        self.assertEqual(await self.plugin.handle_incoming_message(message=message), {"action": "continue"})
        self.assertEqual(message, before)
        self.context.api.call.assert_not_awaited()
        self.plugin._get_or_create_analysis.assert_not_awaited()

    async def test_latest_video_tool_uses_verified_original_message(self):
        await self.remember()
        result = await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2", group_id="2")
        self.assertTrue(result["success"])
        self.context.api.call.assert_awaited_once_with("adapter.napcat.message.get_msg", version="1", message_id=123)
        self.assertIn("QQ聊天视频", result["content"])
        self.assertNotIn("private-signed-token", result["content"])
        self.assertNotIn("链接", result["content"])
        video = self.plugin._get_or_create_analysis.call_args.args[1]
        self.assertEqual(video.source_platform, "qq")
        self.assertEqual(video.canonical_url, "")

    async def test_latest_selection_and_explicit_id(self):
        await self.remember(message_id="123")
        await self.remember(message_id="124")
        self.assertEqual(self.plugin._get_chat_video_authorization("group-2", "").message_id, "124")
        self.assertEqual(self.plugin._get_chat_video_authorization("group-2", "123").message_id, "123")
        self.assertIsNone(self.plugin._get_chat_video_authorization("other-stream", "123"))

    async def test_cross_stream_and_unobserved_ids_fail_before_lookup(self):
        await self.remember()
        for stream, message_id in [("other-stream", "123"), ("group-2", "999"), ("", "123")]:
            result = await guarded_call(self.plugin, "qq_video_summarize", stream_id=stream, message_id=message_id)
            self.assertFalse(result["success"])
        self.context.api.call.assert_not_awaited()

    async def test_cross_group_and_private_sender_fail_before_lookup(self):
        await self.remember()
        result = await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2", group_id="3")
        self.assertFalse(result["success"])
        await self.remember(message_id="124", stream="private-1", group="")
        result = await guarded_call(self.plugin, "qq_video_summarize", stream_id="private-1", user_id="9")
        self.assertFalse(result["success"])
        self.context.api.call.assert_not_awaited()

    async def test_expired_record_and_disabled_switch_do_not_lookup(self):
        auth = await self.remember()
        self.plugin._chat_video_authorizations[("group-2", "123")] = replace(auth, expires_at=time.monotonic() - 1)
        result = await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2")
        self.assertFalse(result["success"])
        await self.remember()
        self.config["chat_video"]["enabled"] = False
        self.plugin.set_plugin_config(self.config)
        self.assertFalse((await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2"))["success"])
        self.context.api.call.assert_not_awaited()

    async def test_whitelist_denial_and_hot_revocation_do_not_lookup(self):
        await self.remember(user="9")
        self.assertFalse((await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2", user_id="1"))["success"])
        await self.remember()
        self.config["access"]["user_whitelist"] = ["9"]
        self.plugin.set_plugin_config(self.config)
        self.assertFalse((await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2"))["success"])
        self.context.api.call.assert_not_awaited()

    async def test_lookup_response_identity_is_checked(self):
        auth = await self.remember()
        for field, value in [("message_id", 999), ("message_type", "private"), ("group_id", 3), ("sender", {"user_id": 9})]:
            response = original()
            response[field] = value
            with self.subTest(field=field), self.assertRaises(ChatVideoError):
                resolve_chat_video(response, auth)

    async def test_private_and_negative_message_id_are_supported(self):
        auth = await self.remember(message_id="-123", stream="private-1", group="")
        self.context.api.call.return_value = original(-123, group=None)
        result = await guarded_call(self.plugin, "qq_video_summarize", stream_id="private-1", user_id="1")
        self.assertTrue(result["success"])
        self.assertEqual(self.context.api.call.call_args.kwargs["message_id"], -123)
        with self.assertRaises(ChatVideoError):
            resolve_chat_video(original(-123), auth)

    async def test_fake_video_text_and_multiple_video_segments_are_rejected(self):
        auth = await self.remember()
        response = original()
        for segments in ["[CQ:video,url=https://example.com]", [{"type": "text", "data": {"text": "[视频]"}}], response["message"] * 2]:
            candidate = {**response, "message": segments}
            with self.subTest(segments=segments), self.assertRaises(ChatVideoError):
                resolve_chat_video(candidate, auth)

    async def test_sdk_direct_detail_and_envelopes_are_supported(self):
        auth = await self.remember()
        for payload in [original(), {"status": "ok", "retcode": 0, "data": original()}, {"success": True, "result": original()}]:
            self.assertEqual(resolve_chat_video(payload, auth).source_platform, "qq")
        for payload in [None, {"success": False, "error": "secret"}, {"retcode": 1, "data": original()}]:
            with self.assertRaises(ChatVideoError):
                resolve_chat_video(payload, auth)

    async def test_missing_file_address_and_metadata_limits_fail_before_model(self):
        await self.remember()
        for changes in [{"url": "file:///etc/passwd"}, {"url": ""}, {"duration": 3601}, {"file_size": 2 * 1024 ** 3}]:
            response = original()
            response["message"][0]["data"].update(changes)
            self.context.api.call.return_value = response
            self.assertFalse((await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2"))["success"])
        self.plugin._get_or_create_analysis.assert_not_awaited()

    async def test_nonqq_and_synthetic_messages_are_not_authorized(self):
        for changes in [{"platform": "web"}, {"message_id": "fake"}, {"message_id": "0"}, {"session_id": ""}]:
            await self.plugin.handle_incoming_message(message={**incoming(), **changes})
        self.assertEqual(self.plugin._chat_video_authorizations, {})

    async def test_adapter_exception_does_not_leak_payload(self):
        await self.remember()
        self.context.api.call.side_effect = RuntimeError("https://gchat.qpic.cn/secret?token=sensitive")
        result = await guarded_call(self.plugin, "qq_video_summarize", stream_id="group-2")
        self.assertFalse(result["success"])
        self.assertNotIn("sensitive", str(result))
        self.assertNotIn("sensitive", str(self.context.logger.mock_calls))

    async def test_cache_identity_is_session_message_and_file_scoped(self):
        auth = await self.remember()
        video = resolve_chat_video(original(), auth)
        other_stream = resolve_chat_video(original(), replace(auth, stream_id="another-stream"))
        other_message = resolve_chat_video(original(124), replace(auth, message_id="124"))
        self.assertNotEqual(video.cache_identity, other_stream.cache_identity)
        self.assertNotEqual(video.cache_identity, other_message.cache_identity)
        rotated = original()
        rotated["message"][0]["data"]["url"] = "https://gchat.qpic.cn/video/test.mp4?token=rotated"
        self.assertEqual(video.cache_identity, resolve_chat_video(rotated, auth).cache_identity)
        before = self.plugin._cache_key(video)
        self.config["access"]["max_duration_seconds"] = 20
        self.plugin.set_plugin_config(self.config)
        self.assertNotEqual(before, self.plugin._cache_key(video))

    async def test_qq_retry_refreshes_adapter_and_redacts_url_in_cache_payload(self):
        auth = await self.remember()
        video = resolve_chat_video(original(), auth)
        self.config["qwen"]["media_transfer_mode"] = "direct_url"
        self.plugin.set_plugin_config(self.config)
        client = MagicMock()
        client.analyze = AsyncMock(side_effect=[
            QwenTimeoutError("request timeout"),
            QwenResult(text=f"内容 {video.sources[0].url}", usage={}, degraded=True),
        ])
        client.close = AsyncMock()
        self.plugin._resolve_video = AsyncMock(side_effect=AssertionError("must never call Bilibili"))
        with patch("maibot_qwen_bilibili.plugin.QwenOmniClient", return_value=client) as factory:
            payload = await self.plugin._run_analysis(video, chat_authorization=auth)
        self.assertEqual(client.analyze.await_count, 2)
        self.context.api.call.assert_awaited_once()
        self.assertEqual(factory.call_args.kwargs["media_transfer_mode"], "auto")
        self.assertEqual(factory.call_args.kwargs["max_duration_seconds"], 3600)
        self.assertNotIn("private-signed-token", str(payload))
        client.close.assert_awaited_once()

    async def test_queue_rechecks_authorization_before_paid_model_call(self):
        auth = await self.remember()
        video = resolve_chat_video(original(), auth)
        self.config["chat_video"]["enabled"] = False
        self.plugin.set_plugin_config(self.config)
        with patch("maibot_qwen_bilibili.plugin.QwenOmniClient") as factory:
            with self.assertRaises(ChatVideoError):
                await self.plugin._run_analysis(video, chat_authorization=auth)
        factory.assert_not_called()

    def test_planner_has_optional_message_id_but_no_url_argument(self):
        components = self.plugin.get_components()
        tool = next(item for item in components if item["name"] == "qq_video_summarize")
        metadata = tool["metadata"]
        self.assertEqual(metadata["visibility"], "visible")
        schema = metadata["parameters_raw"]
        self.assertEqual(set(schema["properties"]), {"message_id"})
        self.assertFalse(schema.get("required"))


if __name__ == "__main__":
    unittest.main()
