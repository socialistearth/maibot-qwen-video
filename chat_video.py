"""Resolve QQ video segments only from a verified, observed inbound message."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .bilibili_client import MediaSource, VideoInfo


class ChatVideoError(RuntimeError):
    """An error whose message contains no private media URL or adapter payload."""


@dataclass(frozen=True, slots=True)
class ChatVideoAuthorization:
    stream_id: str
    message_id: str
    user_id: str
    group_id: str
    is_group: bool
    allowed: bool
    expires_at: float
    sequence: int


def is_video_candidate(message: dict[str, Any], text: str) -> bool:
    """A candidate is only a lookup hint; typed get_msg data is authoritative."""

    if "[视频]" in text:
        return True
    raw = message.get("raw_message")
    return isinstance(raw, list) and any(
        isinstance(segment, dict) and segment.get("type") == "video" for segment in raw
    )


def resolve_chat_video(payload: Any, authorization: ChatVideoAuthorization) -> VideoInfo:
    """Check OneBot identity before extracting the one actual video segment."""

    # SDK 2.x unwraps api.call's host {success, result} response; tolerate that
    # wrapper and a raw OneBot envelope for compatible adapters as well.
    if isinstance(payload, dict) and payload.get("success") is True and isinstance(payload.get("result"), dict):
        payload = payload["result"]
    if not isinstance(payload, dict) or payload.get("status") not in (None, "ok"):
        raise ChatVideoError("适配器未能读取这条QQ消息；消息可能已撤回或过期，请重新发送视频")
    if payload.get("retcode") not in (None, 0, "0") or payload.get("success") is False:
        raise ChatVideoError("适配器读取QQ消息失败，请重新发送视频后重试")
    data = payload.get("data") if "data" in payload else payload
    if not isinstance(data, dict):
        raise ChatVideoError("适配器没有返回有效的QQ原始消息")
    if str(data.get("message_id") or "").strip() != authorization.message_id:
        raise ChatVideoError("适配器返回的消息ID与当前授权消息不一致，已拒绝读取")

    sender = data.get("sender") if isinstance(data.get("sender"), dict) else {}
    sender_id = str(sender.get("user_id") or data.get("user_id") or "").strip()
    if not sender_id or sender_id != authorization.user_id:
        raise ChatVideoError("适配器返回的视频发送者与原始消息不一致，已拒绝读取")
    if data.get("user_id") is not None and str(data["user_id"]).strip() != sender_id:
        raise ChatVideoError("适配器返回了不一致的发送者信息，已拒绝读取")

    message_type = str(data.get("message_type") or "").lower()
    group_id = str(data.get("group_id") or "").strip()
    if authorization.is_group:
        if message_type != "group" or group_id != authorization.group_id:
            raise ChatVideoError("适配器返回的视频不属于当前群聊，已拒绝读取")
    elif message_type != "private" or group_id not in ("", "0"):
        raise ChatVideoError("适配器返回的视频不属于当前私聊，已拒绝读取")

    segments = data.get("message")
    if not isinstance(segments, list):
        raise ChatVideoError("适配器未提供结构化视频消息；文本中的视频标记不能作为视频文件")
    videos = [item for item in segments if isinstance(item, dict) and item.get("type") == "video"]
    if not videos:
        raise ChatVideoError("这条QQ原始消息中没有直接发送的视频文件；请直接发送视频后重试")
    if len(videos) != 1:
        raise ChatVideoError("单条消息包含多个视频，无法确定要识别的文件；请将视频逐条发送")
    video_data = videos[0].get("data")
    if not isinstance(video_data, dict):
        raise ChatVideoError("QQ视频消息缺少有效的文件信息")
    url = str(video_data.get("url") or "").strip()
    try:
        parsed = urlparse(url)
        valid = parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
        valid = valid and parsed.username is None and parsed.password is None
    except ValueError:
        valid = False
    if not valid:
        raise ChatVideoError("QQ视频尚未提供可下载的网络地址，或文件已过期；请重新发送视频")

    file_id = str(video_data.get("file_id") or video_data.get("file") or "video:0")
    identity = hashlib.sha256(
        "\0".join([authorization.stream_id, authorization.message_id, file_id]).encode("utf-8")
    ).hexdigest()
    duration = _nonnegative_number(video_data.get("duration"))
    size = _nonnegative_number(video_data.get("file_size") or video_data.get("size"))
    return VideoInfo(
        original_url="",
        canonical_url="",
        bvid=identity,
        aid=0,
        cid=0,
        page=1,
        part_title="",
        title="QQ聊天视频",
        owner_name="QQ消息发送者",
        description="",
        duration_seconds=duration,
        cover_url="",
        sources=(MediaSource(kind="video", url=url, format="mp4", size_bytes=size),),
        source_platform="qq",
    )


def safe_chat_error(error: Exception) -> str:
    """Adapter/HTTP failures must not expose a signed download URL to Planner."""

    if not isinstance(error, ChatVideoError):
        # Even SDK error text can contain the full upstream response or tokens.
        from .qwen_client import QwenError

        if not isinstance(error, QwenError):
            return f"QQ视频处理失败（{type(error).__name__}）；请重新发送视频或稍后重试"
    text = str(error).strip() or type(error).__name__
    text = re.sub(r"https?://[^\s<>\"']+", "[地址已隐藏]", text, flags=re.IGNORECASE)
    return text[:300]


def _nonnegative_number(value: Any) -> int:
    try:
        number = float(value or 0)
        return math.ceil(number) if math.isfinite(number) and number > 0 else 0
    except (TypeError, ValueError, OverflowError):
        return 0
