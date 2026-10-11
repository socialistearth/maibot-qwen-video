"""Public QQ media destinations and bounded, binary-free MP4 inspection."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import socket
import struct
from pathlib import Path
from typing import Any, BinaryIO, Iterator
from urllib.parse import urlsplit


class MediaValidationError(ValueError):
    """A safe-to-log validation failure; never contains a signed media URL."""


# QQ's traditional CDN and NT multimedia CDN. Match complete DNS suffixes;
# a user-supplied URL outside these hosts is never a chat-video attachment.
_QQ_MEDIA_DOMAINS = ("qq.com", "qq.com.cn", "qpic.cn", "gtimg.cn", "gtimg.com")
_MP4_BRANDS = {b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"dash", b"M4V ", b"MSNV"}


def validate_qq_media_url(url: str) -> str:
    """Reject arbitrary hosts, credentials, non-HTTP schemes and odd ports."""
    if not isinstance(url, str) or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise MediaValidationError("QQ视频下载地址格式无效")
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except (ValueError, TypeError):
        raise MediaValidationError("QQ视频下载地址格式无效") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "\\" in url
        or port not in {None, 80, 443}
        or not any(host == domain or host.endswith(f".{domain}") for domain in _QQ_MEDIA_DOMAINS)
    ):
        raise MediaValidationError("QQ视频下载地址不属于受支持的公开QQ媒体服务")
    return url


def _public_address(address: str) -> bool:
    try:
        value = ipaddress.ip_address(address)
        if isinstance(value, ipaddress.IPv6Address) and value.ipv4_mapped:
            value = value.ipv4_mapped
        return value.is_global and not any((value.is_multicast, value.is_reserved, value.is_unspecified))
    except ValueError:
        return False


class PublicQQResolver:
    """Resolve once into validated IPs actually used by aiohttp's connector.

    A separate session with trust_env=False prevents proxy/DNS rebinding from
    defeating the public-address check while preserving TLS host verification.
    """

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> list[dict[str, Any]]:
        validate_qq_media_url(f"https://{host}:{port or 443}/")
        records = await asyncio.get_running_loop().getaddrinfo(
            host, port, family=family, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
        if not records or any(not _public_address(record[4][0]) for record in records):
            raise MediaValidationError("QQ视频下载域名解析到非公网地址，已拒绝下载")
        return [
            {
                "hostname": host,
                "host": record[4][0],
                "port": record[4][1],
                "family": record[0],
                "proto": record[2],
                "flags": socket.AI_NUMERICHOST,
            }
            for record in records
        ]

    async def close(self) -> None:
        return None


def mp4_duration_seconds(path: Path) -> float:
    """Validate an MP4 movie and return its duration without decoding media.

    All atom bounds are checked, large mdat payloads are skipped by seeking, and
    an atom budget bounds work on malicious tiny-atom files. The movie header
    is mandatory; unknown/zero timescales or durations fail closed.
    """
    try:
        with path.open("rb") as handle:
            return _inspect_mp4(handle, path.stat().st_size)
    except MediaValidationError:
        raise
    except (OSError, ValueError, OverflowError, struct.error):
        raise MediaValidationError("无法读取QQ视频的MP4时长") from None


def _inspect_mp4(handle: BinaryIO, file_size: int) -> float:
    atom_budget = 10000

    def atoms(start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
        nonlocal atom_budget
        offset = start
        while offset < end:
            atom_budget -= 1
            if atom_budget < 0 or end - offset < 8:
                raise MediaValidationError("QQ视频MP4结构损坏或过于复杂")
            handle.seek(offset)
            raw = handle.read(8)
            if len(raw) != 8:
                raise MediaValidationError("QQ视频MP4结构不完整")
            size, kind = struct.unpack(">I4s", raw)
            header_size = 8
            if size == 1:
                raw = handle.read(8)
                if len(raw) != 8:
                    raise MediaValidationError("QQ视频MP4结构不完整")
                size = struct.unpack(">Q", raw)[0]
                header_size = 16
            elif size == 0:
                size = end - offset
            if size < header_size or offset + size > end:
                raise MediaValidationError("QQ视频MP4结构损坏")
            yield kind, offset + header_size, offset + size
            offset += size

    def duration(start: int, end: int, *, movie: bool) -> float:
        handle.seek(start)
        raw = handle.read(min(end - start, 112))
        if not raw or raw[0] not in {0, 1}:
            raise MediaValidationError("无法确认QQ视频的MP4时长")
        version = raw[0]
        required = (112 if version else 100) if movie else (36 if version else 24)
        if len(raw) < required:
            raise MediaValidationError("QQ视频MP4时长信息不完整")
        if version:
            scale, ticks = struct.unpack_from(">IQ", raw, 20)
            unknown = 0xFFFFFFFFFFFFFFFF
        else:
            scale, ticks = struct.unpack_from(">II", raw, 12)
            unknown = 0xFFFFFFFF
        if not scale or not ticks or ticks == unknown:
            raise MediaValidationError("无法确认QQ视频的MP4时长")
        result = ticks / scale
        if not math.isfinite(result) or result <= 0:
            raise MediaValidationError("无法确认QQ视频的MP4时长")
        return result

    has_ftyp = False
    has_media = False
    has_video_track = False
    movie_duration: float | None = None
    track_durations: list[float] = []
    for kind, start, end in atoms(0, file_size):
        if kind == b"ftyp":
            if has_ftyp or end - start < 8 or end - start > 4096 or (end - start) % 4:
                raise MediaValidationError("QQ视频不是有效的MP4文件")
            handle.seek(start)
            brands = handle.read(end - start)
            has_ftyp = any(brand in _MP4_BRANDS for brand in [brands[:4], *[brands[i:i + 4] for i in range(8, len(brands), 4)]])
            if not has_ftyp:
                raise MediaValidationError("QQ视频暂仅支持MP4格式")
        elif kind == b"mdat":
            has_media = has_media or end > start
        elif kind == b"moov":
            for child_kind, child_start, child_end in atoms(start, end):
                if child_kind == b"mvhd":
                    if movie_duration is not None:
                        raise MediaValidationError("QQ视频含有重复的MP4时长信息")
                    movie_duration = duration(child_start, child_end, movie=True)
                elif child_kind == b"trak":
                    for track_kind, track_start, track_end in atoms(child_start, child_end):
                        if track_kind != b"mdia":
                            continue
                        for media_kind, media_start, media_end in atoms(track_start, track_end):
                            if media_kind == b"hdlr":
                                if media_end - media_start < 24:
                                    raise MediaValidationError("QQ视频MP4轨道信息不完整")
                                handle.seek(media_start + 8)
                                has_video_track = has_video_track or handle.read(4) == b"vide"
                            elif media_kind == b"mdhd":
                                track_durations.append(duration(media_start, media_end, movie=False))
    if not has_ftyp or not has_media or not has_video_track:
        raise MediaValidationError("QQ视频不是包含视频轨道的有效MP4文件")
    if movie_duration is None:
        raise MediaValidationError("无法确认QQ视频的MP4时长")
    return max(movie_duration, *track_durations) if track_durations else movie_duration
