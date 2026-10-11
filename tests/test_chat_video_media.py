from __future__ import annotations

import asyncio
import base64
import socket
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from maibot_qwen_bilibili.bilibili_client import MediaSource, VideoInfo
from maibot_qwen_bilibili.media_guard import (
    MediaValidationError,
    PublicQQResolver,
    mp4_duration_seconds,
    validate_qq_media_url,
)
from maibot_qwen_bilibili.qwen_client import QwenError, QwenOmniClient, QwenTransientError


def atom(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I4s", len(data) + 8, kind) + data


def mp4(*, seconds: float = 5, version: int = 0, timescale: int = 1000, handler: bytes = b"vide", track_seconds: float | None = None) -> bytes:
    if version:
        mvhd = b"\x01\0\0\0" + struct.pack(">QQIQ", 0, 0, timescale, int(seconds * 1000)) + b"\0" * 80
    else:
        mvhd = b"\0" * 4 + struct.pack(">IIII", 0, 0, timescale, int(seconds * 1000)) + b"\0" * 80
    media = atom(b"hdlr", b"\0" * 8 + handler + b"\0" * 12)
    if track_seconds is not None:
        media += atom(b"mdhd", b"\0" * 4 + struct.pack(">IIII", 0, 0, 1000, int(track_seconds * 1000)) + b"\0" * 4)
    return (
        atom(b"ftyp", b"mp42\0\0\0\0isommp42")
        + atom(b"mdat", b"example-video-payload")
        + atom(b"moov", atom(b"mvhd", mvhd) + atom(b"trak", atom(b"mdia", media)))
    )


def video(**kwargs: object) -> VideoInfo:
    fields = dict(
        original_url="qq-message:123", canonical_url="qq-message:123", bvid="attachment-hash", aid=0,
        cid=0, page=1, part_title="", title="", owner_name="", description="", duration_seconds=0,
        cover_url="", sources=(MediaSource("video", "https://gchat.qpic.cn/file.mp4?token=private-test-token", "mp4"),),
        source_platform="qq",
    )
    fields.update(kwargs)
    return VideoInfo(**fields)


def client(**kwargs: object) -> QwenOmniClient:
    options = dict(api_key="test", api_base="https://dashscope.aliyuncs.com/compatible-mode/v1", model="test", timeout_seconds=60, max_output_tokens=512)
    options.update(kwargs)
    return QwenOmniClient(**options)


class MP4Tests(unittest.TestCase):
    def inspect(self, payload: bytes) -> float:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.mp4"
            path.write_bytes(payload)
            return mp4_duration_seconds(path)

    def test_mvhd_versions_and_fractional_duration(self) -> None:
        for version in (0, 1):
            with self.subTest(version=version):
                self.assertEqual(self.inspect(mp4(seconds=5.125, version=version)), 5.125)

    def test_longer_track_duration_cannot_bypass_limit(self) -> None:
        self.assertEqual(self.inspect(mp4(seconds=5, track_seconds=10)), 10)

    def test_invalid_containers_and_unknown_durations_are_rejected(self) -> None:
        for payload in (
            b"<html>expired download link</html>", b"", mp4()[:-1], mp4(handler=b"soun"),
            mp4(seconds=0), mp4(timescale=0), mp4().replace(b"mvhd", b"free"),
            mp4().replace(b"mp42", b"avif").replace(b"isom", b"avif"),
            atom(b"ftyp", b"isom\0\0\0\0") + struct.pack(">I4s", 0xFFFFFFFF, b"moov"),
        ):
            with self.subTest(payload=payload[:16]):
                with self.assertRaises(MediaValidationError):
                    self.inspect(payload)

    def test_missing_video_data_rejected(self) -> None:
        with self.assertRaises(MediaValidationError):
            self.inspect(mp4().replace(b"mdat", b"free"))


class MediaURLTests(unittest.IsolatedAsyncioTestCase):
    def test_qq_cdn_urls_accepted(self) -> None:
        for host in ("gchat.qpic.cn", "multimedia.nt.qq.com.cn", "video.qq.com", "a.gtimg.cn"):
            self.assertEqual(validate_qq_media_url(f"https://{host}/download?token=x"), f"https://{host}/download?token=x")

    def test_private_external_disguised_and_local_urls_rejected(self) -> None:
        for url in (
            "http://127.0.0.1/a.mp4", "http://10.0.0.1/a.mp4", "file:///root/a.mp4", "C:/a.mp4",
            "https://qq.com.evil.example/a.mp4", "https://evilqq.com/a.mp4", "https://user:pass@qq.com/a.mp4",
            "https://gchat.qpic.cn:8080/a.mp4", "https://gchat.qpic.cn/a.mp4#x", "https://gchat.qpic.cn/a\r\nx",
            "https://gchat.qpic.cn\\@example.com/a.mp4", "https://[::1]/a.mp4",
        ):
            with self.subTest(url=url):
                with self.assertRaises(MediaValidationError):
                    validate_qq_media_url(url)

    async def test_resolver_rejects_any_private_dns_answer(self) -> None:
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "224.0.0.1", "::1", "::ffff:127.0.0.1"):
            with self.subTest(address=address):
                records = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))]
                with patch.object(asyncio.get_running_loop(), "getaddrinfo", AsyncMock(return_value=records)):
                    with self.assertRaises(MediaValidationError):
                        await PublicQQResolver().resolve("gchat.qpic.cn", 443)

    async def test_resolver_returns_validated_address_to_connector(self) -> None:
        records = [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("1.1.1.1", 443))]
        with patch.object(asyncio.get_running_loop(), "getaddrinfo", AsyncMock(return_value=records)):
            addresses = await PublicQQResolver().resolve("gchat.qpic.cn", 443)
        self.assertEqual(addresses[0]["host"], "1.1.1.1")
        self.assertEqual(addresses[0]["hostname"], "gchat.qpic.cn")


class FakeResponse:
    def __init__(self, data: bytes = b"", *, status: int = 200, headers: dict[str, str] | None = None):
        self.data = data
        self.status = status
        self.headers = headers or {}
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def iter_chunked(self, size):
        for start in range(0, len(self.data), size):
            yield self.data[start:start + size]

    async def read(self, size):
        return self.data[:size]


class FakeSession:
    def __init__(self, *responses: FakeResponse):
        self.responses = list(responses)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


class ChatMediaTransferTests(unittest.IsolatedAsyncioTestCase):
    async def download(self, session: FakeSession, *, instance: QwenOmniClient | None = None, source: MediaSource | None = None) -> bytes:
        instance = instance or client()
        instance._get_chat_media_session = AsyncMock(return_value=session)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "media.mp4"
            await instance._download_media_source(video(), source or video().sources[0], path)
            return path.read_bytes()

    async def test_qq_download_does_not_send_bilibili_referer(self) -> None:
        session = FakeSession(FakeResponse(mp4()))
        self.assertEqual(await self.download(session), mp4())
        self.assertNotIn("Referer", session.requests[0][1]["headers"])
        self.assertFalse(session.requests[0][1]["allow_redirects"])

    async def test_each_redirect_is_checked_before_fetch(self) -> None:
        session = FakeSession(FakeResponse(status=302, headers={"Location": "http://127.0.0.1/secret"}))
        with self.assertRaisesRegex(QwenError, "公开QQ"):
            await self.download(session)
        self.assertEqual(len(session.requests), 1)

    async def test_valid_relative_redirect(self) -> None:
        session = FakeSession(FakeResponse(status=302, headers={"Location": "/real.mp4"}), FakeResponse(mp4()))
        self.assertEqual(await self.download(session), mp4())
        self.assertEqual(session.requests[1][0], "https://gchat.qpic.cn/real.mp4")

    async def test_metadata_declared_and_streamed_size_limits(self) -> None:
        instances = client(max_media_download_mb=1), client(max_media_download_mb=1), client(max_media_download_mb=1)
        with self.assertRaisesRegex(QwenError, "超过"):
            await self.download(FakeSession(), instance=instances[0], source=replace(video().sources[0], size_bytes=1048577))
        with self.assertRaisesRegex(QwenError, "超过"):
            await self.download(FakeSession(FakeResponse(headers={"Content-Length": "1048577"})), instance=instances[1])
        with self.assertRaisesRegex(QwenError, "超过"):
            await self.download(FakeSession(FakeResponse(b"x" * 1048577)), instance=instances[2])

    async def test_truncated_or_empty_download_fails(self) -> None:
        for response in (FakeResponse(), FakeResponse(b"123", headers={"Content-Length": "4"})):
            with self.assertRaises(QwenTransientError):
                await self.download(FakeSession(response))

    async def test_expired_nt_video_requests_resend_without_retry_or_url_leak(self) -> None:
        payloads = (
            b'{"retcode":-5503007,"retmsg":"secret-signature"}',
            b'{"retcode":1,"retmsg":"signature expired https://qpic.cn/?rkey=secret-signature"}',
        )
        for payload in payloads:
            with self.subTest(payload=payload[:24]):
                with self.assertRaises(QwenError) as caught:
                    await self.download(FakeSession(FakeResponse(payload, status=400)))
                self.assertNotIsInstance(caught.exception, QwenTransientError)
                self.assertEqual(str(caught.exception), "QQ视频下载地址已过期，请重新发送视频后再试")
                self.assertNotIn("secret-signature", str(caught.exception))

    async def test_unavailable_media_not_retried_but_server_failures_are(self) -> None:
        for status in (400, 401, 403, 404, 410):
            with self.subTest(status=status):
                with self.assertRaisesRegex(QwenError, "重新发送") as caught:
                    await self.download(FakeSession(FakeResponse(b"not JSON", status=status)))
                self.assertNotIsInstance(caught.exception, QwenTransientError)
        for status in (429, 500, 503):
            with self.subTest(status=status):
                with self.assertRaises(QwenTransientError):
                    await self.download(FakeSession(FakeResponse(status=status)))

    async def test_direct_url_still_downloads_qq_then_sends_base64(self) -> None:
        instance = client(media_transfer_mode="direct_url")
        session = FakeSession(FakeResponse(mp4()))
        instance._get_chat_media_session = AsyncMock(return_value=session)
        with tempfile.TemporaryDirectory() as directory:
            prepared, uses_oss = await instance._prepare_media_source(video(), video().sources[0], index=0, temporary_dir=Path(directory))
        self.assertFalse(uses_oss)
        self.assertEqual(base64.b64decode(prepared.split(",", 1)[1]), mp4())
        self.assertEqual(len(session.requests), 1)

    async def test_actual_duration_and_invalid_container_checked_before_upload(self) -> None:
        for data in (mp4(seconds=60.001), b"<html>expired</html>", mp4(seconds=0)):
            instance = client(max_duration_seconds=60, media_transfer_mode="temporary_oss")
            instance._get_chat_media_session = AsyncMock(return_value=FakeSession(FakeResponse(data)))
            instance._upload_temporary_oss = AsyncMock(return_value="oss://unexpected")
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(QwenError):
                    await instance._prepare_media_source(video(), video().sources[0], index=0, temporary_dir=Path(directory))
            instance._upload_temporary_oss.assert_not_awaited()

    async def test_temporary_oss_receives_original_whole_video(self) -> None:
        instance = client(media_transfer_mode="temporary_oss")
        instance._get_chat_media_session = AsyncMock(return_value=FakeSession(FakeResponse(mp4())))

        async def upload(path: Path, *, content_type: str):
            self.assertEqual(path.read_bytes(), mp4())
            self.assertEqual(content_type, "video/mp4")
            return "oss://temporary/video.mp4"

        instance._upload_temporary_oss = upload
        with tempfile.TemporaryDirectory() as directory:
            result = await instance._prepare_media_source(video(), video().sources[0], index=0, temporary_dir=Path(directory))
        self.assertEqual(result, ("oss://temporary/video.mp4", True))

    async def test_existing_bilibili_direct_url_and_cache_identity_unchanged(self) -> None:
        instance = client(media_transfer_mode="direct_url")
        bili = video(source_platform="bilibili", bvid="BVtest", cid=42)
        self.assertEqual(bili.cache_identity, "BVtest:42")
        self.assertEqual(video().cache_identity, "qq:attachment-hash:0")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(await instance._prepare_media_source(bili, bili.sources[0], index=0, temporary_dir=Path(directory)), (bili.sources[0].url, False))

    def test_prompt_omits_fake_metadata_and_signed_url(self) -> None:
        prompt = QwenOmniClient._build_prompt(video(), "理解整个视频", degraded=False)
        self.assertIn("QQ聊天", prompt)
        self.assertNotIn("B站", prompt)
        self.assertNotIn("private-test-token", prompt)
        self.assertNotIn("UP主：", prompt)

    def test_upstream_error_redacts_media_urls(self) -> None:
        message = QwenOmniClient._safe_api_message('{"error":{"message":"fetch https://qpic.cn/download?token=private-test-token failed"}}')
        self.assertNotIn("private-test-token", message)
        self.assertIn("failed", message)


if __name__ == "__main__":
    unittest.main()
