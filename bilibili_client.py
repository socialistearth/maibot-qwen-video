"""Bilibili link resolution without external binaries or Python packages."""

from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import string
import time
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import aiohttp


_LONG_LINK_RE = re.compile(
    r"(?:(?:https?://)?(?:(?:www|m|space)\.)?bilibili\.com/video/(?:BV[0-9A-Za-z]+|av\d+)"
    r"(?:[/?#][0-9A-Za-z._~:/?#@!$&()*+,;=%-]*)?)",
    re.IGNORECASE,
)
_SHORT_LINK_RE = re.compile(
    r"(?:(?:https?://)?(?:b23\.tv|bili2233\.cn|bili23\.cn)/[0-9A-Za-z_-]+"
    r"(?:[/?#][0-9A-Za-z._~:/?#@!$&()*+,;=%-]*)?)",
    re.IGNORECASE,
)
_VIDEO_ID_RE = re.compile(r"/(BV[0-9A-Za-z]+|av\d+)(?:/|$)", re.IGNORECASE)

_QUALITY_IDS = {
    "360p": 16,
    "480p": 32,
    "720p": 64,
    "1080p": 80,
}

# getMixinKey() permutation used by Bilibili's web client. Keeping it local
# avoids adding yt-dlp or another runtime dependency to the MaiBot image.
_WBI_MIXIN_KEY_ENC_TAB = (
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
)
_WBI_KEY_TTL_SECONDS = 30.0


class BilibiliError(RuntimeError):
    """A safe-to-log Bilibili resolution error."""


@dataclass(frozen=True, slots=True)
class MediaSource:
    """One remotely fetchable media input."""

    kind: str
    url: str
    format: str = ""
    size_bytes: int = 0


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Resolved metadata and media streams for one supported video source."""

    original_url: str
    canonical_url: str
    bvid: str
    aid: int
    cid: int
    page: int
    part_title: str
    title: str
    owner_name: str
    description: str
    duration_seconds: int
    cover_url: str
    sources: tuple[MediaSource, ...]
    source_platform: str = "bilibili"

    @property
    def cache_identity(self) -> str:
        if self.source_platform != "bilibili":
            return f"{self.source_platform}:{self.bvid}:{self.cid}"
        return f"{self.bvid}:{self.cid}"


def extract_bilibili_links(text: str) -> list[str]:
    """Extract unique long and short Bilibili links in message order."""

    matches: list[tuple[int, str]] = []
    for pattern in (_LONG_LINK_RE, _SHORT_LINK_RE):
        matches.extend((match.start(), match.group(0)) for match in pattern.finditer(text or ""))
    matches.sort(key=lambda item: item[0])

    links: list[str] = []
    seen: set[str] = set()
    for _, raw_link in matches:
        link = raw_link.rstrip(".,，。!！?？;；:：)）")
        if not link.lower().startswith(("http://", "https://")):
            link = f"https://{link}"
        if link not in seen:
            seen.add(link)
            links.append(link)
    return links


class BilibiliClient:
    """Resolve public Bilibili pages to Qwen-consumable media URLs."""

    def __init__(self, *, timeout_seconds: float = 20.0, cookie: str = "") -> None:
        self._timeout_seconds = max(float(timeout_seconds), 3.0)
        self._cookie = self._normalize_cookie(cookie)
        self._cookies = self._parse_cookie(self._cookie)
        self._device_cookie_ready = False
        self._wbi_key = ""
        self._wbi_key_timestamp = 0.0
        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def resolve(
        self,
        link: str,
        *,
        preferred_quality: str = "480p",
        max_segments: int = 16,
    ) -> VideoInfo:
        resolved_url = await self._resolve_short_link(link)
        video_id, page = self._parse_video_url(resolved_url)
        self._ensure_device_cookie()

        params = {"bvid": video_id} if video_id.lower().startswith("bv") else {"aid": video_id[2:]}
        view = await self._api_json("https://api.bilibili.com/x/web-interface/view", params=params)

        pages = view.get("pages") if isinstance(view.get("pages"), list) else []
        if not pages:
            raise BilibiliError("B站接口没有返回可播放分P")
        page = min(max(page, 1), len(pages))
        selected_page = pages[page - 1]
        cid = self._as_int(selected_page.get("cid"))
        if not cid:
            raise BilibiliError("B站接口没有返回视频 cid")

        bvid = str(view.get("bvid") or video_id).strip()
        aid = self._as_int(view.get("aid"))
        quality_id = _QUALITY_IDS.get(preferred_quality, _QUALITY_IDS["480p"])
        sources = await self._resolve_media(
            bvid=bvid,
            aid=aid,
            cid=cid,
            page=page,
            quality_id=quality_id,
            max_segments=max(1, int(max_segments)),
        )

        duration = self._as_int(selected_page.get("duration")) or self._as_int(view.get("duration"))
        canonical_url = f"https://www.bilibili.com/video/{bvid}"
        if page > 1:
            canonical_url += f"?p={page}"
        owner = view.get("owner") if isinstance(view.get("owner"), dict) else {}
        cover_url = str(view.get("pic") or "").strip()
        if cover_url.startswith("//"):
            cover_url = f"https:{cover_url}"

        return VideoInfo(
            original_url=link,
            canonical_url=canonical_url,
            bvid=bvid,
            aid=aid,
            cid=cid,
            page=page,
            part_title=str(selected_page.get("part") or "").strip(),
            title=str(view.get("title") or bvid).strip(),
            owner_name=str(owner.get("name") or "未知UP主").strip(),
            description=str(view.get("desc") or "").strip(),
            duration_seconds=duration,
            cover_url=cover_url,
            sources=tuple(sources),
        )

    async def _resolve_media(
        self,
        *,
        bvid: str,
        aid: int,
        cid: int,
        page: int,
        quality_id: int,
        max_segments: int,
    ) -> list[MediaSource]:
        del aid  # bvid + cid uniquely identify the selected page for playurl.
        common_params: dict[str, Any] = {
            "bvid": bvid,
            "cid": cid,
            "qn": quality_id,
            "fnver": 0,
            "fourk": 0,
            "high_quality": 1,
            **self._browser_fingerprint_params(),
        }
        referer = self._canonical_url(bvid, page)
        wbi_error: BilibiliError | None = None

        try:
            # A signed durl response normally contains picture and sound in one
            # file and therefore remains the best no-ffmpeg path.
            play = await self._wbi_playurl({**common_params, "fnval": 1}, referer=referer)
            if sources := self._sources_from_play_data(play, quality_id, max_segments):
                return sources

            # Qwen3.5-Omni accepts separate video and audio URLs in one request,
            # so DASH remains a binary-free fallback.
            play = await self._wbi_playurl({**common_params, "fnval": 4048}, referer=referer)
            if sources := self._sources_from_play_data(play, quality_id, max_segments):
                return sources
            wbi_error = BilibiliError("WBI播放接口没有返回可供模型读取的视频流")
        except BilibiliError as exc:
            wbi_error = exc

        # Bilibili's video page often embeds the same signed playback payload.
        # This path is deliberately independent of api.bilibili.com, so an API
        # HTTP 412 does not automatically make the whole plugin fail.
        try:
            embedded_play = await self._embedded_playinfo(bvid=bvid, page=page)
            if sources := self._sources_from_play_data(embedded_play, quality_id, max_segments):
                return sources
            page_error = "网页内嵌播放信息中没有可用视频流"
        except BilibiliError as exc:
            page_error = str(exc)

        api_detail = str(wbi_error or "未知错误")
        raise BilibiliError(f"B站取流失败（WBI：{api_detail}；网页回退：{page_error}）")

    async def _wbi_playurl(self, params: dict[str, Any], *, referer: str) -> dict[str, Any]:
        mixin_key = await self._get_wbi_key()
        signed_params = self._sign_wbi_params(params, mixin_key=mixin_key)
        return await self._api_json(
            "https://api.bilibili.com/x/player/wbi/playurl",
            params=signed_params,
            referer=referer,
        )

    async def _get_wbi_key(self) -> str:
        now = time.time()
        if self._wbi_key and now < self._wbi_key_timestamp + _WBI_KEY_TTL_SECONDS:
            return self._wbi_key

        nav = await self._api_json(
            "https://api.bilibili.com/x/web-interface/nav",
            params={},
            accepted_codes={-101},  # Anonymous nav still includes wbi_img.
        )
        wbi_img = nav.get("wbi_img") if isinstance(nav.get("wbi_img"), dict) else {}
        img_key = self._wbi_resource_key(wbi_img.get("img_url"))
        sub_key = self._wbi_resource_key(wbi_img.get("sub_url"))
        mixin_key = self._mixin_key(img_key, sub_key)
        if not mixin_key:
            raise BilibiliError("B站导航接口没有返回有效的 WBI 密钥")

        self._wbi_key = mixin_key
        self._wbi_key_timestamp = now
        return mixin_key

    @staticmethod
    def _wbi_resource_key(value: Any) -> str:
        path = urlparse(str(value or "")).path
        return PurePosixPath(path).stem

    @staticmethod
    def _mixin_key(img_key: str, sub_key: str) -> str:
        lookup = f"{img_key}{sub_key}"
        if len(lookup) <= max(_WBI_MIXIN_KEY_ENC_TAB):
            return ""
        return "".join(lookup[index] for index in _WBI_MIXIN_KEY_ENC_TAB)[:32]

    @staticmethod
    def _sign_wbi_params(
        params: dict[str, Any],
        *,
        mixin_key: str,
        timestamp: int | None = None,
    ) -> dict[str, str]:
        signed: dict[str, Any] = dict(params)
        signed["wts"] = round(time.time()) if timestamp is None else int(timestamp)
        normalized = {
            key: "".join(char for char in str(value) if char not in "!'()*")
            for key, value in sorted(signed.items())
        }
        query = urlencode(normalized)
        normalized["w_rid"] = hashlib.md5(f"{query}{mixin_key}".encode()).hexdigest()
        return normalized

    @staticmethod
    def _browser_fingerprint_params() -> dict[str, str]:
        dimensions = random.choices(
            ((1920, 1080), (1366, 768), (1536, 864), (1280, 720), (2560, 1440), (1440, 900), (1600, 900)),
            weights=(18, 18, 17, 8, 7, 5, 5),
        )[0]

        def encoded_random_text(minimum: int, maximum: int) -> str:
            raw = "".join(random.choices(string.printable, k=random.randint(minimum, maximum))).encode()
            return base64.b64encode(raw).decode().rstrip("=")

        def dimensions_fingerprint(width: int, height: int) -> list[int]:
            offset = random.randrange(114)
            return [2 * width + 2 * height + 3 * offset, 4 * width - height + offset, offset]

        scroll_top = random.randint(0, 100)
        offset = random.randrange(514)
        scroll_fingerprint = [3 * scroll_top + offset, 4 * scroll_top + 2 * offset, offset]
        return {
            "dm_img_list": "[]",
            "dm_img_str": encoded_random_text(16, 64),
            "dm_cover_img_str": encoded_random_text(32, 128),
            "dm_img_inter": json.dumps(
                {
                    "ds": [],
                    "wh": dimensions_fingerprint(*dimensions),
                    "of": scroll_fingerprint,
                },
                separators=(",", ":"),
            ),
        }

    @classmethod
    def _sources_from_play_data(
        cls,
        play: dict[str, Any],
        quality_id: int,
        max_segments: int,
    ) -> list[MediaSource]:
        durl = play.get("durl") if isinstance(play.get("durl"), list) else []
        combined_sources = [
            MediaSource(
                kind="video",
                url=cls._pick_url(item),
                format=cls._guess_video_format(cls._pick_url(item), str(play.get("format") or "")),
                size_bytes=cls._as_int(item.get("size")),
            )
            for item in durl[:max_segments]
            if isinstance(item, dict) and cls._pick_url(item)
        ]
        if combined_sources:
            return combined_sources

        dash = play.get("dash") if isinstance(play.get("dash"), dict) else {}
        video_track = cls._choose_dash_video(dash.get("video"), quality_id)

        audio_candidates = list(dash.get("audio") or []) if isinstance(dash.get("audio"), list) else []
        dolby = dash.get("dolby") if isinstance(dash.get("dolby"), dict) else {}
        if isinstance(dolby.get("audio"), list):
            audio_candidates.extend(dolby["audio"])
        flac = dash.get("flac") if isinstance(dash.get("flac"), dict) else {}
        if isinstance(flac.get("audio"), dict):
            audio_candidates.append(flac["audio"])
        audio_track = cls._choose_dash_audio(audio_candidates)

        if not video_track:
            return []
        sources = [MediaSource(kind="video", url=cls._pick_url(video_track), format="mp4")]
        if audio_track and cls._pick_url(audio_track):
            sources.append(MediaSource(kind="audio", url=cls._pick_url(audio_track), format="m4a"))
        return sources

    async def _embedded_playinfo(self, *, bvid: str, page: int) -> dict[str, Any]:
        url = self._canonical_url(bvid, page)
        session = await self._get_session()
        try:
            async with session.get(url, headers=self._headers(referer=url, html=True)) as response:
                self._update_response_cookies(response)
                if response.status != 200:
                    raise BilibiliError(f"B站视频页返回 HTTP {response.status}")
                html = await response.text(errors="replace")
        except BilibiliError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise BilibiliError(f"访问B站视频页失败：{type(exc).__name__}") from exc

        play = self._extract_embedded_playinfo(html)
        if play is None:
            raise BilibiliError("B站视频页没有嵌入播放信息")
        return play

    @staticmethod
    def _extract_embedded_playinfo(html: str) -> dict[str, Any] | None:
        marker = "window.__playinfo__"
        search_from = 0
        decoder = json.JSONDecoder()
        while True:
            marker_at = html.find(marker, search_from)
            if marker_at < 0:
                return None
            equals_at = html.find("=", marker_at + len(marker))
            if equals_at < 0:
                return None
            candidate = html[equals_at + 1 :].lstrip()
            try:
                payload, _ = decoder.raw_decode(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                search_from = marker_at + len(marker)
                continue
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, dict):
                    return data
            search_from = marker_at + len(marker)

    @staticmethod
    def _canonical_url(bvid: str, page: int) -> str:
        url = f"https://www.bilibili.com/video/{bvid}"
        return f"{url}?p={page}" if page > 1 else url

    async def _api_json(
        self,
        url: str,
        *,
        params: dict[str, Any],
        referer: str = "https://www.bilibili.com/",
        accepted_codes: set[int] | None = None,
    ) -> dict[str, Any]:
        session = await self._get_session()
        headers = self._headers(referer=referer)
        try:
            async with session.get(url, params=params, headers=headers) as response:
                self._update_response_cookies(response)
                if response.status != 200:
                    raise BilibiliError(f"B站接口返回 HTTP {response.status}")
                payload = await response.json(content_type=None)
        except BilibiliError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise BilibiliError(f"访问B站接口失败：{type(exc).__name__}") from exc

        if not isinstance(payload, dict):
            raise BilibiliError("B站接口返回了无法识别的数据")
        code = self._as_int(payload.get("code"))
        if code != 0 and code not in (accepted_codes or set()):
            message = str(payload.get("message") or payload.get("msg") or "未知错误").strip()
            raise BilibiliError(f"B站接口拒绝请求（{code}：{message}）")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise BilibiliError("B站接口响应中缺少 data")
        return data

    async def _resolve_short_link(self, link: str) -> str:
        parsed = urlparse(link)
        host = (parsed.hostname or "").lower()
        if host not in {"b23.tv", "bili2233.cn", "bili23.cn"}:
            return link

        session = await self._get_session()
        current_url = link
        redirect_statuses = {301, 302, 303, 307, 308}
        short_hosts = {"b23.tv", "bili2233.cn", "bili23.cn"}
        try:
            for _redirect_index in range(5):
                # Do not automatically fetch the final www.bilibili.com page.
                # b23.tv itself normally returns a usable Location header, while
                # following that redirect can trigger Bilibili's HTTP 412 risk
                # control before the signed API pipeline even begins.
                async with session.get(
                    current_url,
                    headers=self._headers(html=True),
                    allow_redirects=False,
                ) as response:
                    self._update_response_cookies(response)
                    if response.status >= 400:
                        raise BilibiliError(f"B站短链返回 HTTP {response.status}")
                    if response.status not in redirect_statuses:
                        raise BilibiliError("B站短链没有返回可用的跳转地址")
                    location = str(response.headers.get("Location") or "").strip()
                    if not location:
                        raise BilibiliError("B站短链响应缺少 Location 跳转地址")

                next_url = urljoin(current_url, location)
                next_host = (urlparse(next_url).hostname or "").lower()
                if next_host == "bilibili.com" or next_host.endswith(".bilibili.com"):
                    return next_url
                if next_host not in short_hosts:
                    raise BilibiliError("B站短链跳转到了非 bilibili.com 域名，已拒绝处理")
                current_url = next_url
        except BilibiliError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise BilibiliError(f"展开B站短链失败：{type(exc).__name__}") from exc

        raise BilibiliError("B站短链跳转次数过多")

    @staticmethod
    def _parse_video_url(link: str) -> tuple[str, int]:
        parsed = urlparse(link)
        host = (parsed.hostname or "").lower()
        if not (host == "bilibili.com" or host.endswith(".bilibili.com")):
            raise BilibiliError("不是受支持的B站视频链接")
        match = _VIDEO_ID_RE.search(parsed.path)
        if not match:
            raise BilibiliError("链接中没有找到 BV/av 号")
        page_text = (parse_qs(parsed.query).get("p") or ["1"])[0]
        try:
            page = max(int(page_text), 1)
        except (TypeError, ValueError):
            page = 1
        return match.group(1), page

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout, trust_env=True)
        return self._session

    def _headers(
        self,
        *,
        referer: str = "https://www.bilibili.com/",
        html: bool = False,
    ) -> dict[str, str]:
        headers = {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
                if html
                else "application/json,text/plain,*/*"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Referer": referer,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
            ),
        }
        if self._cookies:
            headers["Cookie"] = "; ".join(f"{key}={value}" for key, value in self._cookies.items())
        return headers

    def _ensure_device_cookie(self) -> None:
        if self._device_cookie_ready:
            return
        # Bilibili treats a browser without buvid3 as a suspicious client. The
        # web value is not an account credential and can be generated per plugin
        # process when the administrator did not supply one.
        self._cookies.setdefault("buvid3", f"{uuid.uuid4()}infoc")
        self._cookies.setdefault("b_nut", str(round(time.time())))
        self._device_cookie_ready = True

    def _update_response_cookies(self, response: Any) -> None:
        response_cookies = getattr(response, "cookies", None)
        if not response_cookies:
            return
        for key, morsel in response_cookies.items():
            value = getattr(morsel, "value", "")
            if value:
                self._cookies[str(key)] = str(value)

    @staticmethod
    def _normalize_cookie(cookie: str) -> str:
        value = str(cookie or "").strip()
        if not value:
            return ""
        if "=" not in value:
            return f"SESSDATA={value}"
        return value

    @staticmethod
    def _parse_cookie(cookie: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for item in str(cookie or "").split(";"):
            key, separator, value = item.strip().partition("=")
            if separator and key:
                parsed[key] = value
        return parsed

    @staticmethod
    def _pick_url(item: dict[str, Any]) -> str:
        direct = str(item.get("url") or item.get("baseUrl") or item.get("base_url") or "").strip()
        if direct:
            return direct
        backups = item.get("backupUrl") or item.get("backup_url") or []
        if isinstance(backups, list) and backups:
            return str(backups[0] or "").strip()
        return ""

    @staticmethod
    def _choose_dash_video(raw_tracks: Any, quality_id: int) -> dict[str, Any] | None:
        if not isinstance(raw_tracks, list):
            return None
        tracks = [item for item in raw_tracks if isinstance(item, dict) and BilibiliClient._pick_url(item)]
        if not tracks:
            return None
        eligible = [item for item in tracks if BilibiliClient._as_int(item.get("id")) <= quality_id] or tracks
        # Prefer AVC (codecid 7) for decoder compatibility, then the best resolution/bandwidth.
        return max(
            eligible,
            key=lambda item: (
                BilibiliClient._as_int(item.get("id")),
                1 if BilibiliClient._as_int(item.get("codecid")) == 7 else 0,
                BilibiliClient._as_int(item.get("bandwidth")),
            ),
        )

    @staticmethod
    def _choose_dash_audio(raw_tracks: Any) -> dict[str, Any] | None:
        if not isinstance(raw_tracks, list):
            return None
        tracks = [item for item in raw_tracks if isinstance(item, dict) and BilibiliClient._pick_url(item)]
        if not tracks:
            return None
        return max(tracks, key=lambda item: BilibiliClient._as_int(item.get("bandwidth")))

    @staticmethod
    def _guess_video_format(url: str, api_format: str) -> str:
        suffix = PurePosixPath(urlparse(url).path).suffix.lower().lstrip(".")
        if suffix in {"mp4", "flv", "mkv", "mov", "avi", "wmv"}:
            return suffix
        normalized = api_format.lower()
        return "flv" if "flv" in normalized else "mp4"

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
