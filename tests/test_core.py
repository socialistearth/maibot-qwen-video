from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from maibot_qwen_bilibili.tests.guard_helpers import guarded_call
from pathlib import Path

try:
    import aiohttp  # noqa: F401
    HAS_AIOHTTP = True
except ModuleNotFoundError:
    # The Codex validation runtime is intentionally minimal. MaiBot 1.1.2 itself
    # declares aiohttp>=3.12.14; these pure unit tests do not perform HTTP calls.
    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.ClientError = type("ClientError", (Exception,), {})
    aiohttp_stub.ClientConnectionError = type("ClientConnectionError", (Exception,), {})
    aiohttp_stub.ServerDisconnectedError = type("ServerDisconnectedError", (Exception,), {})
    aiohttp_stub.ClientSession = object
    aiohttp_stub.ClientTimeout = object
    aiohttp_stub.payload = types.SimpleNamespace(Payload=object)
    sys.modules["aiohttp"] = aiohttp_stub
    HAS_AIOHTTP = False

from maibot_qwen_bilibili.access_control import is_allowed
from maibot_qwen_bilibili.bilibili_client import (
    BilibiliClient,
    BilibiliError,
    extract_bilibili_links,
)
from maibot_qwen_bilibili.cache_store import AnalysisCache
from maibot_qwen_bilibili.plugin import QwenBilibiliPlugin
from maibot_qwen_bilibili.qwen_client import QwenOmniClient, _OssMultipartPayload, _delta_text


class LinkTests(unittest.TestCase):
    def test_extracts_long_short_and_deduplicates(self) -> None:
        text = (
            "先看 https://www.bilibili.com/video/BV1Ab411c7dE?p=2，"
            "再看 b23.tv/abc123；重复 https://www.bilibili.com/video/BV1Ab411c7dE?p=2"
        )
        self.assertEqual(
            extract_bilibili_links(text),
            [
                "https://www.bilibili.com/video/BV1Ab411c7dE?p=2",
                "https://b23.tv/abc123",
            ],
        )

    def test_parses_page(self) -> None:
        self.assertEqual(
            BilibiliClient._parse_video_url("https://www.bilibili.com/video/BV1Ab411c7dE?p=3"),
            ("BV1Ab411c7dE", 3),
        )

    def test_extracts_reported_link_with_share_query(self) -> None:
        link = (
            "https://www.bilibili.com/video/BV1oiui6dEZh/"
            "?share_source=copy_web&vd_source=test-fixture"
        )
        self.assertEqual(extract_bilibili_links(link), [link])
        self.assertEqual(BilibiliClient._parse_video_url(link), ("BV1oiui6dEZh", 1))

    def test_rejects_external_host(self) -> None:
        with self.assertRaises(Exception):
            BilibiliClient._parse_video_url("https://example.com/video/BV1Ab411c7dE")


class BilibiliPlaybackTests(unittest.IsolatedAsyncioTestCase):
    def test_wbi_signing_matches_reference_vector(self) -> None:
        img_key = "7cd084941338484aae1ad9425b84077c"
        sub_key = "4932caff0ff746eab6f01bf08b70ac45"
        mixin_key = BilibiliClient._mixin_key(img_key, sub_key)
        self.assertEqual(mixin_key, "ea1db124af3c7062474693fa704f4ff8")
        signed = BilibiliClient._sign_wbi_params(
            {"foo": "114", "bar": "514", "baz": 1919810},
            mixin_key=mixin_key,
            timestamp=1702204169,
        )
        self.assertEqual(signed["wts"], "1702204169")
        self.assertEqual(signed["w_rid"], "6149fdadf571698ca7e6a567265cd0ee")

    def test_extracts_page_embedded_playinfo(self) -> None:
        html = (
            '<html><script>window.__playinfo__ = {"code":0,"data":'
            '{"durl":[{"url":"https://cdn.example/video.mp4","size":42}]}};'
            "</script></html>"
        )
        play = BilibiliClient._extract_embedded_playinfo(html)
        self.assertEqual(play["durl"][0]["size"], 42)

    def test_generates_device_cookie_without_overwriting_admin_cookie(self) -> None:
        client = BilibiliClient(cookie="SESSDATA=secret; buvid3=admin-value")
        client._ensure_device_cookie()
        self.assertEqual(client._cookies["SESSDATA"], "secret")
        self.assertEqual(client._cookies["buvid3"], "admin-value")
        self.assertIn("b_nut", client._cookies)

    async def test_http_412_uses_embedded_page_fallback(self) -> None:
        client = BilibiliClient()

        async def fail_wbi(params: dict[str, object], *, referer: str) -> dict[str, object]:
            del params, referer
            raise BilibiliError("B站接口返回 HTTP 412")

        async def embedded(*, bvid: str, page: int) -> dict[str, object]:
            self.assertEqual((bvid, page), ("BV1oiui6dEZh", 1))
            return {"durl": [{"url": "https://cdn.example/full.mp4", "size": 128}]}

        client._wbi_playurl = fail_wbi  # type: ignore[method-assign]
        client._embedded_playinfo = embedded  # type: ignore[method-assign]
        sources = await client._resolve_media(
            bvid="BV1oiui6dEZh",
            aid=1,
            cid=2,
            page=1,
            quality_id=32,
            max_segments=16,
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://cdn.example/full.mp4")

    async def test_short_link_reads_location_without_fetching_final_page(self) -> None:
        client = BilibiliClient()

        class FakeResponse:
            status = 302
            headers = {"Location": "https://www.bilibili.com/video/BV1Qug96wEky?p=1"}
            cookies: dict[str, object] = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback

        class FakeSession:
            closed = False

            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def get(self, url: str, *, headers: dict[str, str], allow_redirects: bool):
                del headers
                self.calls.append((url, allow_redirects))
                return FakeResponse()

        session = FakeSession()
        client._session = session  # type: ignore[assignment]
        resolved = await client._resolve_short_link("https://b23.tv/VqztWoh")
        self.assertEqual(resolved, "https://www.bilibili.com/video/BV1Qug96wEky?p=1")
        self.assertEqual(session.calls, [("https://b23.tv/VqztWoh", False)])


class AccessTests(unittest.TestCase):
    def test_empty_is_denied_by_default(self) -> None:
        self.assertFalse(
            is_allowed(
                user_id="1",
                group_id="2",
                is_group=True,
                user_whitelist=[],
                group_whitelist=[],
                mode="any",
                allow_when_empty=False,
                enable_group=True,
                enable_private=True,
            )
        )

    def test_any_all_semantics(self) -> None:
        common = dict(
            user_id="1",
            group_id="2",
            is_group=True,
            user_whitelist=["1"],
            group_whitelist=["9"],
            allow_when_empty=False,
            enable_group=True,
            enable_private=True,
        )
        self.assertTrue(is_allowed(mode="any", **common))
        self.assertFalse(is_allowed(mode="all", **common))

    def test_private_ignores_group_list(self) -> None:
        self.assertTrue(
            is_allowed(
                user_id="1",
                group_id="",
                is_group=False,
                user_whitelist=["1"],
                group_whitelist=["9"],
                mode="all",
                allow_when_empty=False,
                enable_group=True,
                enable_private=True,
            )
        )


class QwenParsingTests(unittest.TestCase):
    def test_delta_text_variants(self) -> None:
        self.assertEqual(_delta_text("abc"), "abc")
        self.assertEqual(_delta_text([{"text": "a"}, {"content": "b"}]), "ab")

    def test_prepared_media_url_replaces_bilibili_url(self) -> None:
        source = types.SimpleNamespace(kind="video", url="https://cdn.example/video.mp4", format="mp4")
        content = QwenOmniClient._source_to_content(
            source,
            prepared_url="oss://dashscope-instant/test/video.mp4",
            fps=1.0,
            max_pixels=655360,
        )
        self.assertEqual(content["video_url"]["url"], "oss://dashscope-instant/test/video.mp4")


@unittest.skipUnless(HAS_AIOHTTP, "requires the aiohttp package included by MaiBot")
class QwenTransferTests(unittest.IsolatedAsyncioTestCase):
    async def test_oss_multipart_payload_has_known_size_and_file_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "media-00.mp4"
            media.write_bytes(b"video-bytes")
            payload = _OssMultipartPayload(
                media,
                fields=[("key", "temporary/media-00.mp4"), ("success_action_status", "200")],
                content_type="video/mp4",
            )

            class Writer:
                def __init__(self) -> None:
                    self.chunks: list[bytes] = []

                async def write(self, data: bytes) -> None:
                    self.chunks.append(data)

            writer = Writer()
            await payload.write(writer)
            body = b"".join(writer.chunks)
            self.assertEqual(payload.size, len(body))
            self.assertIn(b'name="file"; filename="media-00.mp4"', body)
            self.assertIn(b"video-bytes", body)


class PlannerToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_inbound_link_can_be_analyzed_by_tool(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["allow_when_empty"] = True
        plugin.set_plugin_config(config)

        async def fake_process(link: str) -> str:
            self.assertIn("BV1Ab411c7dE", link)
            return "【B站视频完整音视频理解】\n模型分析：测试结果"

        plugin._process_link = fake_process  # type: ignore[method-assign]
        message = {
            "session_id": "qq-group-2",
            "processed_plain_text": (
                "看看 https://www.bilibili.com/video/BV1Ab411c7dE/"
                "?share_source=copy_web&vd_source=test-fixture"
            ),
            "message_info": {
                "user_info": {"user_id": "1"},
                "group_info": {"group_id": "2"},
            },
        }
        hook_result = await plugin.handle_incoming_message(message=message)
        self.assertEqual(hook_result, {"action": "continue"})
        self.assertNotIn("仅供 MaiBot 内部理解", message["processed_plain_text"])

        tool_result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://www.bilibili.com/video/BV1Ab411c7dE",
            stream_id="qq-group-2",
            group_id="2",
        )
        self.assertTrue(tool_result["success"])
        self.assertIn("仅供 MaiBot 内部理解", tool_result["content"])
        self.assertIn("测试结果", tool_result["content"])
        self.assertIn("无需再用网页搜索", tool_result["content"])

    async def test_group_all_mode_uses_real_sender_from_inbound_hook(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["user_whitelist"] = ["1"]
        config["access"]["group_whitelist"] = ["2"]
        config["access"]["whitelist_mode"] = "all"
        plugin.set_plugin_config(config)

        async def fake_process(link: str) -> str:
            del link
            return "完整视频结果"

        plugin._process_link = fake_process  # type: ignore[method-assign]
        await plugin.handle_incoming_message(
            message={
                "session_id": "qq-group-2",
                "processed_plain_text": "https://www.bilibili.com/video/BV1Ab411c7dE",
                "message_info": {
                    "user_info": {"user_id": "1"},
                    "group_info": {"group_id": "2"},
                },
            }
        )
        # MaiBot 1.1.2's group ChatStream has no user_id. The tool must still
        # enforce the user whitelist using the sender captured by the hook.
        result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://www.bilibili.com/video/BV1Ab411c7dE",
            stream_id="qq-group-2",
            user_id="",
            group_id="2",
        )
        self.assertTrue(result["success"])

    async def test_denied_link_never_runs_paid_pipeline(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["user_whitelist"] = ["allowed-user"]
        plugin.set_plugin_config(config)
        process_called = False

        async def fake_process(link: str) -> str:
            nonlocal process_called
            process_called = True
            return link

        plugin._process_link = fake_process  # type: ignore[method-assign]
        message = {
            "session_id": "private-denied",
            "processed_plain_text": "https://www.bilibili.com/video/BV1Ab411c7dE",
            "message_info": {"user_info": {"user_id": "denied-user"}},
        }
        # Avoid requiring a Runner context merely for the expected debug log.
        plugin._remember_tool_authorizations(
            stream_id=message["session_id"],
            links=extract_bilibili_links(message["processed_plain_text"]),
            user_id="denied-user",
            group_id="",
            is_group=False,
            allowed=False,
        )
        result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://www.bilibili.com/video/BV1Ab411c7dE",
            stream_id="private-denied",
            user_id="denied-user",
        )
        self.assertFalse(result["success"])
        self.assertIn("未通过", result["error"])
        self.assertFalse(process_called)

    async def test_tool_requires_link_observed_in_current_session(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["allow_when_empty"] = True
        plugin.set_plugin_config(config)
        result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://www.bilibili.com/video/BV1Ab411c7dE",
            stream_id="missing-session",
        )
        self.assertFalse(result["success"])
        self.assertIn("入站授权记录", result["error"])

    async def test_planner_expanded_short_link_uses_original_inbound_url(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["allow_when_empty"] = True
        plugin.set_plugin_config(config)
        processed_links: list[str] = []

        async def fake_process(link: str) -> str:
            processed_links.append(link)
            return "短链视频结果"

        plugin._process_link = fake_process  # type: ignore[method-assign]
        await plugin.handle_incoming_message(
            message={
                "session_id": "short-link-session",
                "processed_plain_text": "帮我看看 https://b23.tv/VqztWoh",
                "message_info": {"user_info": {"user_id": "1"}},
            }
        )
        result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://www.bilibili.com/video/BV1Ab411c7dE",
            stream_id="short-link-session",
            user_id="1",
        )
        self.assertTrue(result["success"])
        self.assertEqual(processed_links, ["https://b23.tv/VqztWoh"])

    async def test_exact_short_link_authorization_still_works(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["allow_when_empty"] = True
        plugin.set_plugin_config(config)

        async def fake_process(link: str) -> str:
            self.assertEqual(link, "https://b23.tv/VqztWoh")
            return "短链视频结果"

        plugin._process_link = fake_process  # type: ignore[method-assign]
        await plugin.handle_incoming_message(
            message={
                "session_id": "short-link-session",
                "processed_plain_text": "https://b23.tv/VqztWoh",
                "message_info": {"user_info": {"user_id": "1"}},
            }
        )
        result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://b23.tv/VqztWoh",
            stream_id="short-link-session",
            user_id="1",
        )
        self.assertTrue(result["success"])

    async def test_multiple_short_links_never_fall_back_ambiguously(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["allow_when_empty"] = True
        config["access"]["max_links_per_message"] = 2
        plugin.set_plugin_config(config)
        process_called = False

        async def fake_process(link: str) -> str:
            nonlocal process_called
            process_called = True
            return link

        plugin._process_link = fake_process  # type: ignore[method-assign]
        await plugin.handle_incoming_message(
            message={
                "session_id": "ambiguous-short-links",
                "processed_plain_text": "https://b23.tv/VqztWoh https://b23.tv/Another1",
                "message_info": {"user_info": {"user_id": "1"}},
            }
        )
        result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://www.bilibili.com/video/BV1Ab411c7dE",
            stream_id="ambiguous-short-links",
            user_id="1",
        )
        self.assertFalse(result["success"])
        self.assertIn("入站授权记录", result["error"])
        self.assertFalse(process_called)

    async def test_denied_short_link_fallback_cannot_run_paid_pipeline(self) -> None:
        plugin = QwenBilibiliPlugin()
        config = plugin.get_default_config()
        config["access"]["user_whitelist"] = ["allowed-user"]
        plugin.set_plugin_config(config)
        process_called = False

        async def fake_process(link: str) -> str:
            nonlocal process_called
            process_called = True
            return link

        plugin._process_link = fake_process  # type: ignore[method-assign]
        plugin._remember_tool_authorizations(
            stream_id="denied-short-link",
            links=["https://b23.tv/VqztWoh"],
            user_id="denied-user",
            group_id="",
            is_group=False,
            allowed=False,
        )
        result = await guarded_call(plugin, "bilibili_video_summarize", 
            url="https://www.bilibili.com/video/BV1Ab411c7dE",
            stream_id="denied-short-link",
            user_id="denied-user",
        )
        self.assertFalse(result["success"])
        self.assertIn("未通过", result["error"])
        self.assertFalse(process_called)

    def test_tool_is_directly_visible_to_planner(self) -> None:
        plugin = QwenBilibiliPlugin()
        components = plugin.get_components()
        tool = next(item for item in components if item["name"] == "bilibili_video_summarize")
        metadata = tool["metadata"]
        self.assertEqual(tool["type"], "TOOL")
        self.assertEqual(metadata["visibility"], "visible")
        self.assertEqual(metadata["timeout_ms"], 1_500_000)
        self.assertEqual(metadata["parameters_raw"]["required"], ["url"])
        self.assertIn("不要用网页搜索代替", metadata["description"])


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_trip_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = AnalysisCache(Path(directory) / "cache.sqlite3")
            await cache.open()
            await cache.put("a", {"text": "A"}, max_entries=1)
            self.assertEqual(await cache.get("a", ttl_hours=1), {"text": "A"})
            await asyncio.sleep(0.01)
            await cache.put("b", {"text": "B"}, max_entries=1)
            self.assertIsNone(await cache.get("a", ttl_hours=1))
            self.assertEqual(await cache.count(), 1)
            await cache.close()


if __name__ == "__main__":
    unittest.main()
