"""Minimal asynchronous Qwen-Omni OpenAI-compatible streaming client."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp

from .bilibili_client import MediaSource, VideoInfo
from .media_guard import MediaValidationError, PublicQQResolver, mp4_duration_seconds, validate_qq_media_url


_DATA_URL_LIMIT_BYTES = 10 * 1024 * 1024
_TEMP_OSS_LIMIT_BYTES = 1024 * 1024 * 1024


class _OssMultipartPayload(aiohttp.payload.Payload):
    """Streaming multipart body compatible with Aliyun OSS POST forms."""

    def __init__(
        self,
        file_path: Path,
        *,
        fields: list[tuple[str, str]],
        content_type: str,
    ) -> None:
        self._file_path = file_path
        boundary = uuid.uuid4().hex
        chunks: list[bytes] = []
        for name, value in fields:
            if any(char in name or char in value for char in ("\r", "\n")):
                raise ValueError("multipart field contains a newline")
            chunks.append(
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode("utf-8")
            )
        chunks.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{file_path.name}"\r\nContent-Type: {content_type}\r\n\r\n'
            ).encode("utf-8")
        )
        self._preamble = b"".join(chunks)
        self._closing = f"\r\n--{boundary}--\r\n".encode("ascii")
        super().__init__(None, content_type=f"multipart/form-data; boundary={boundary}")
        self._size = len(self._preamble) + file_path.stat().st_size + len(self._closing)

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        del encoding, errors
        raise TypeError("streaming multipart payload cannot be decoded as text")

    async def write(self, writer: Any) -> None:
        await writer.write(self._preamble)
        with self._file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                await writer.write(chunk)
        await writer.write(self._closing)


class QwenError(RuntimeError):
    """Base Qwen request error."""


class QwenTimeoutError(QwenError):
    """The request exceeded the configured wall-clock timeout."""


class QwenTransientError(QwenError):
    """An error that is normally safe to retry once."""


@dataclass(frozen=True, slots=True)
class QwenResult:
    text: str
    usage: dict[str, Any]
    degraded: bool = False


def _delta_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return ""


class QwenOmniClient:
    """Call Qwen-Omni directly with aiohttp; no new image dependency is needed."""

    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        temperature: float = 0.2,
        media_transfer_mode: str = "auto",
        max_media_download_mb: int = 1024,
        max_duration_seconds: int = 3600,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.api_base = str(api_base or "").strip().rstrip("/")
        self.model = str(model or "").strip()
        self.timeout_seconds = max(float(timeout_seconds), 10.0)
        self.max_output_tokens = max(int(max_output_tokens), 256)
        self.temperature = min(max(float(temperature), 0.0), 2.0)
        normalized_mode = str(media_transfer_mode or "auto").strip().lower()
        self.media_transfer_mode = normalized_mode if normalized_mode in {"auto", "temporary_oss", "direct_url"} else "auto"
        self.max_media_download_bytes = min(max(int(max_media_download_mb), 1) * 1024 * 1024, _TEMP_OSS_LIMIT_BYTES)
        self.max_duration_seconds = max(int(max_duration_seconds), 1)
        self._session: aiohttp.ClientSession | None = None
        self._chat_media_session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        if self._chat_media_session is not None and not self._chat_media_session.closed:
            await self._chat_media_session.close()
        self._chat_media_session = None

    async def analyze(
        self,
        video: VideoInfo,
        *,
        prompt: str,
        fps: float,
        max_pixels: int,
        degraded: bool = False,
    ) -> QwenResult:
        if not self.api_key:
            raise QwenError("尚未配置百炼 API Key")
        if not self.api_base.startswith("https://"):
            raise QwenError("Qwen API 地址必须使用 https://")
        if not self.model:
            raise QwenError("尚未配置 Qwen 模型名称")

        try:
            return await asyncio.wait_for(
                self._analyze_with_media(
                    video,
                    prompt=prompt,
                    fps=fps,
                    max_pixels=max_pixels,
                    degraded=degraded,
                ),
                timeout=self.timeout_seconds,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise QwenTimeoutError(f"Qwen 请求超过 {self.timeout_seconds:g} 秒") from exc

    async def _analyze_with_media(
        self,
        video: VideoInfo,
        *,
        prompt: str,
        fps: float,
        max_pixels: int,
        degraded: bool,
    ) -> QwenResult:
        content: list[dict[str, Any]] = []
        uses_temporary_oss = False
        with tempfile.TemporaryDirectory(prefix="maibot-qwen-bilibili-") as temporary_dir:
            for index, source in enumerate(video.sources):
                prepared_url, uses_oss = await self._prepare_media_source(
                    video,
                    source,
                    index=index,
                    temporary_dir=Path(temporary_dir),
                )
                uses_temporary_oss = uses_temporary_oss or uses_oss
                content.append(
                    self._source_to_content(
                        source,
                        prepared_url=prepared_url,
                        fps=fps,
                        max_pixels=max_pixels,
                    )
                )

            content.append({"type": "text", "text": self._build_prompt(video, prompt, degraded=degraded)})
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是严谨的音视频内容理解器。你的输出会作为另一聊天模型的内部上下文。"
                            "只陈述能从视频画面、音轨、字幕和提供的元数据中得到的信息，不虚构。"
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                "modalities": ["text"],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
            return await self._stream_request(
                payload,
                degraded=degraded,
                resolve_temporary_oss=uses_temporary_oss,
            )

    async def _prepare_media_source(
        self,
        video: VideoInfo,
        source: MediaSource,
        *,
        index: int,
        temporary_dir: Path,
    ) -> tuple[str, bool]:
        is_chat_video = video.source_platform == "qq"
        if self.media_transfer_mode == "direct_url" and not is_chat_video:
            return source.url, False

        suffix = "mp4" if is_chat_video else self._safe_media_suffix(source)
        local_path = temporary_dir / f"media-{index:02d}.{suffix}"
        await self._download_media_source(video, source, local_path)
        if is_chat_video:
            try:
                actual_duration = mp4_duration_seconds(local_path)
            except MediaValidationError as exc:
                raise QwenError(str(exc)) from None
            if actual_duration > self.max_duration_seconds:
                raise QwenError(
                    f"QQ视频时长 {actual_duration:g} 秒，超过允许上限 {self.max_duration_seconds} 秒"
                )

        data_url_prefix = f"data:{self._media_content_type(source, suffix)};base64,"
        encoded_size = 4 * ((local_path.stat().st_size + 2) // 3) + len(data_url_prefix)
        transfer_mode = "auto" if is_chat_video and self.media_transfer_mode == "direct_url" else self.media_transfer_mode
        if transfer_mode == "auto" and encoded_size < _DATA_URL_LIMIT_BYTES:
            encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
            return f"{data_url_prefix}{encoded}", False

        oss_url = await self._upload_temporary_oss(local_path, content_type=self._media_content_type(source, suffix))
        return oss_url, True

    async def _download_media_source(self, video: VideoInfo, source: MediaSource, destination: Path) -> None:
        if source.size_bytes and source.size_bytes > self.max_media_download_bytes:
            raise QwenError(
                f"媒体文件 {source.size_bytes / 1024 / 1024:.1f} MB，超过临时传输上限 "
                f"{self.max_media_download_bytes / 1024 / 1024:.0f} MB"
            )
        if video.source_platform == "qq":
            await self._download_chat_media(source, destination)
            return

        session = await self._get_session()
        headers = {
            "Referer": video.canonical_url,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
            ),
        }
        try:
            async with session.get(source.url, headers=headers, allow_redirects=True) as response:
                if response.status >= 400:
                    raise QwenTransientError(f"下载B站媒体失败：HTTP {response.status}")
                declared_size = int(response.headers.get("Content-Length") or 0)
                if declared_size and declared_size > self.max_media_download_bytes:
                    raise QwenError(
                        f"媒体文件 {declared_size / 1024 / 1024:.1f} MB，超过临时传输上限 "
                        f"{self.max_media_download_bytes / 1024 / 1024:.0f} MB"
                    )
                downloaded = 0
                with destination.open("wb") as handle:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > self.max_media_download_bytes:
                            raise QwenError(
                                f"媒体文件超过临时传输上限 {self.max_media_download_bytes / 1024 / 1024:.0f} MB"
                            )
                        handle.write(chunk)
                if downloaded == 0:
                    raise QwenTransientError("下载B站媒体失败：响应为空")
        except (QwenError, QwenTransientError):
            raise
        except (aiohttp.ClientError, OSError) as exc:
            raise QwenTransientError(f"下载B站媒体失败：{type(exc).__name__}") from exc

    async def _download_chat_media(self, source: MediaSource, destination: Path) -> None:
        """Download only public QQ CDN media, validating each redirect hop."""
        try:
            current_url = validate_qq_media_url(source.url)
            session = await self._get_chat_media_session()
            for _ in range(6):
                async with session.get(
                    current_url,
                    headers={"User-Agent": "MaiBot-ChatVideo/1.0", "Accept-Encoding": "identity"},
                    allow_redirects=False,
                ) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = str(response.headers.get("Location") or "").strip()
                        if not location:
                            raise QwenError("QQ视频下载跳转地址为空")
                        current_url = validate_qq_media_url(urljoin(current_url, location))
                        continue
                    if response.status != 200:
                        if response.status == 400:
                            # QQ uses a small JSON error for expired NT media
                            # signatures. Inspect only a bounded body and never
                            # echo its content, which could contain signed URLs.
                            raw_error = await response.content.read(2048)
                            try:
                                error_data = json.loads(raw_error)
                            except (ValueError, UnicodeDecodeError):
                                error_data = {}
                            if isinstance(error_data, dict) and (
                                str(error_data.get("retcode")) == "-5503007"
                                or "expired" in str(error_data.get("retmsg") or "").lower()
                            ):
                                raise QwenError("QQ视频下载地址已过期，请重新发送视频后再试")
                        if response.status in {400, 401, 403, 404, 410}:
                            raise QwenError(
                                f"QQ视频下载地址无效或已过期（HTTP {response.status}），请重新发送视频后再试"
                            )
                        error_type = QwenTransientError if response.status == 429 or response.status >= 500 else QwenError
                        raise error_type(f"下载QQ视频失败：HTTP {response.status}")
                    try:
                        declared_size = int(response.headers.get("Content-Length") or 0)
                    except (TypeError, ValueError):
                        raise QwenError("QQ视频下载响应中的文件大小无效") from None
                    if declared_size < 0:
                        raise QwenError("QQ视频下载响应中的文件大小无效")
                    if declared_size > self.max_media_download_bytes:
                        raise QwenError(
                            f"QQ视频文件 {declared_size / 1024 / 1024:.1f} MB，超过临时传输上限 "
                            f"{self.max_media_download_bytes / 1024 / 1024:.0f} MB"
                        )
                    downloaded = 0
                    with destination.open("wb") as handle:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            downloaded += len(chunk)
                            if downloaded > self.max_media_download_bytes:
                                raise QwenError(
                                    f"QQ视频文件超过临时传输上限 {self.max_media_download_bytes / 1024 / 1024:.0f} MB"
                                )
                            handle.write(chunk)
                    if not downloaded:
                        raise QwenTransientError("下载QQ视频失败：响应为空")
                    if declared_size and downloaded != declared_size:
                        raise QwenTransientError("下载QQ视频失败：文件未完整传输")
                    return
            raise QwenError("QQ视频下载跳转次数过多")
        except MediaValidationError as exc:
            raise QwenError(str(exc)) from None
        except (aiohttp.ClientError, OSError) as exc:
            # aiohttp exceptions may include the full signed CDN URL. Keep them
            # out of both the message and chained traceback.
            raise QwenTransientError(f"下载QQ视频失败：{type(exc).__name__}") from None

    async def _get_chat_media_session(self) -> aiohttp.ClientSession:
        if self._chat_media_session is None or self._chat_media_session.closed:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=self.timeout_seconds)
            connector = aiohttp.TCPConnector(resolver=PublicQQResolver())
            self._chat_media_session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                trust_env=False,
                auto_decompress=False,
                cookie_jar=aiohttp.DummyCookieJar(),
            )
        return self._chat_media_session

    async def _upload_temporary_oss(self, file_path: Path, *, content_type: str) -> str:
        session = await self._get_session()
        policy_endpoint = self._temporary_upload_endpoint()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with session.get(
                policy_endpoint,
                params={"action": "getPolicy", "model": self.model},
                headers=headers,
            ) as response:
                payload = await response.json(content_type=None)
                if response.status >= 400:
                    message = self._safe_api_message(json.dumps(payload, ensure_ascii=False))
                    error = QwenTransientError if response.status == 429 or response.status >= 500 else QwenError
                    raise error(f"获取百炼临时上传凭证失败：HTTP {response.status}：{message}")
        except (QwenError, QwenTransientError):
            raise
        except (aiohttp.ClientError, ValueError) as exc:
            raise QwenTransientError(f"获取百炼临时上传凭证失败：{type(exc).__name__}") from exc

        policy = payload.get("data") if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else {}
        required = (
            "upload_dir",
            "oss_access_key_id",
            "signature",
            "policy",
            "x_oss_object_acl",
            "x_oss_forbid_overwrite",
            "upload_host",
        )
        if any(not policy.get(field) for field in required):
            raise QwenError("百炼临时上传凭证缺少必要字段")

        upload_host = str(policy["upload_host"])
        parsed_host = urlparse(upload_host)
        if parsed_host.scheme != "https" or not (parsed_host.hostname or "").endswith(".aliyuncs.com"):
            raise QwenError("百炼返回了不受信任的临时上传地址")
        object_key = f"{str(policy['upload_dir']).rstrip('/')}/{file_path.name}"
        fields = [
            ("OSSAccessKeyId", str(policy["oss_access_key_id"])),
            ("Signature", str(policy["signature"])),
            ("policy", str(policy["policy"])),
            ("x-oss-object-acl", str(policy["x_oss_object_acl"])),
            ("x-oss-forbid-overwrite", str(policy["x_oss_forbid_overwrite"])),
            ("key", object_key),
            ("success_action_status", "200"),
        ]

        try:
            upload_payload = _OssMultipartPayload(file_path, fields=fields, content_type=content_type)
            async with session.post(upload_host, data=upload_payload) as response:
                if response.status != 200:
                    error = QwenTransientError if response.status == 429 or response.status >= 500 else QwenError
                    raise error(f"上传媒体到百炼临时空间失败：HTTP {response.status}")
        except (QwenError, QwenTransientError):
            raise
        except (aiohttp.ClientError, OSError, ValueError) as exc:
            raise QwenTransientError(f"上传媒体到百炼临时空间失败：{type(exc).__name__}") from exc
        return f"oss://{object_key}"

    def _temporary_upload_endpoint(self) -> str:
        host = (urlparse(self.api_base).hostname or "").lower()
        if "dashscope-intl" in host or ".ap-southeast-1.maas.aliyuncs.com" in host:
            return "https://dashscope-intl.aliyuncs.com/api/v1/uploads"
        if "dashscope.aliyuncs.com" in host or ".cn-beijing.maas.aliyuncs.com" in host:
            return "https://dashscope.aliyuncs.com/api/v1/uploads"
        raise QwenError("无法根据 Qwen API 地址判断百炼临时上传地域")

    async def _stream_request(
        self,
        payload: dict[str, Any],
        *,
        degraded: bool,
        resolve_temporary_oss: bool = False,
    ) -> QwenResult:
        session = await self._get_session()
        endpoint = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if resolve_temporary_oss:
            headers["X-DashScope-OssResourceResolve"] = "enable"
        try:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                if response.status >= 400:
                    body = await response.text()
                    message = self._safe_api_message(body)
                    if response.status == 429 or response.status >= 500:
                        raise QwenTransientError(f"Qwen API HTTP {response.status}：{message}")
                    raise QwenError(f"Qwen API HTTP {response.status}：{message}")

                chunks: list[str] = []
                usage: dict[str, Any] = {}
                saw_sse = False
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    saw_sse = True
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta")
                        if isinstance(delta, dict):
                            chunks.append(_delta_text(delta.get("content")))

                text = "".join(chunks).strip()
                if not saw_sse:
                    raise QwenError("Qwen API 没有返回预期的流式响应")
                if not text:
                    raise QwenError("Qwen API 返回为空")
                return QwenResult(text=text, usage=usage, degraded=degraded)
        except (QwenError, QwenTransientError):
            raise
        except (aiohttp.ClientConnectionError, aiohttp.ServerDisconnectedError) as exc:
            raise QwenTransientError(f"Qwen 网络连接中断：{type(exc).__name__}") from exc
        except aiohttp.ClientError as exc:
            raise QwenTransientError(f"Qwen 网络请求失败：{type(exc).__name__}") from exc

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # The outer asyncio.wait_for owns the total deadline. A generous socket
            # read timeout still prevents an indefinitely stalled TCP connection.
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=self.timeout_seconds)
            # Honor HTTP_PROXY/HTTPS_PROXY already configured for the MaiBot
            # container; this does not require any deployment or adapter change.
            self._session = aiohttp.ClientSession(timeout=timeout, trust_env=True)
        return self._session

    @staticmethod
    def _source_to_content(
        source: MediaSource,
        *,
        prepared_url: str | None = None,
        fps: float,
        max_pixels: int,
    ) -> dict[str, Any]:
        media_url = prepared_url or source.url
        if source.kind == "audio":
            return {
                "type": "input_audio",
                "input_audio": {"data": media_url, "format": source.format or "m4a"},
            }
        return {
            "type": "video_url",
            "video_url": {"url": media_url},
            "fps": min(max(float(fps), 0.1), 10.0),
            "max_pixels": max(int(max_pixels), 4096),
        }

    @staticmethod
    def _safe_media_suffix(source: MediaSource) -> str:
        candidate = str(source.format or "").strip().lower()
        allowed = {"mp4", "flv", "mkv", "mov", "avi", "wmv", "m4a", "aac", "mp3", "wav"}
        if candidate in allowed:
            return candidate
        return "m4a" if source.kind == "audio" else "mp4"

    @staticmethod
    def _media_content_type(source: MediaSource, suffix: str) -> str:
        if source.kind == "audio":
            return {
                "m4a": "audio/mp4",
                "aac": "audio/aac",
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
            }.get(suffix, "audio/mp4")
        return {
            "flv": "video/x-flv",
            "mkv": "video/x-matroska",
            "mov": "video/quicktime",
            "avi": "video/x-msvideo",
            "wmv": "video/x-ms-wmv",
        }.get(suffix, "video/mp4")

    @staticmethod
    def _build_prompt(video: VideoInfo, prompt: str, *, degraded: bool) -> str:
        description = video.description or "（未提供）"
        if len(description) > 1000:
            description = f"{description[:1000]}…"
        retry_note = (
            "\n本次是超时后的低视觉分辨率重试；视频时间范围仍须从头到尾完整理解。"
            if degraded
            else ""
        )
        if video.source_platform == "qq":
            return (
                "来源：QQ聊天中直接发送的视频附件。\n"
                "请完整理解视频从头到尾的画面、声音、对白和字幕。"
                "该附件未提供经过核实的标题、作者或简介，请勿推断这些信息。"
                f"{retry_note}\n\n任务要求：\n{prompt.strip()}"
            )
        return (
            f"B站标题：{video.title}\n"
            f"UP主：{video.owner_name}\n"
            f"分P：P{video.page} {video.part_title or video.title}\n"
            f"时长：{video.duration_seconds} 秒\n"
            f"简介：{description}\n"
            f"媒体文件按出现顺序提供；若画面和音频是分离轨道，应将它们视作同一视频。{retry_note}\n\n"
            f"任务要求：\n{prompt.strip()}"
        )

    @staticmethod
    def _safe_api_message(body: str) -> str:
        text = (body or "").strip()
        try:
            payload = json.loads(text)
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                text = str(error.get("message") or error.get("code") or "请求失败")
        except json.JSONDecodeError:
            pass
        # Errors may echo request media URLs. Do not leak signed CDN/OSS URLs
        # or embedded base64 payloads into MaiBot logs/context.
        text = re.sub(r"(?:https?://|oss://|data:)[^\s\"'<>]+", "[媒体地址已隐藏]", text)
        return text[:500] or "请求失败"
