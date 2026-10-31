import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from platformdirs import user_data_dir

_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def video_id(url: str) -> str | None:
    if not _youtube(url):
        return None
    parsed = urlparse(url.strip())
    host = _host(parsed.hostname or "")
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate if _ID.fullmatch(candidate) else None
    query = parse_qs(parsed.query).get("v", [""])[0]
    if _ID.fullmatch(query):
        return query
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"} and _ID.fullmatch(parts[1]):
        return parts[1]
    return None


def resolve_audio(path: str, fetch: Callable[[str], Path]) -> Path:
    text = path.strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return Path(text).expanduser()
    if video_id(text) is None:
        if _youtube(text):
            raise ValueError("This YouTube link has no video.")
        raise ValueError("Only a YouTube link works. Pass a file path for other audio.")
    return fetch(text)


def download_youtube(url: str) -> Path:
    found = video_id(url)
    if found is None:
        raise ValueError("This YouTube link has no video.")
    folder = Path(user_data_dir("nemotron-transcribe-mcp", ensure_exists=True)) / "youtube" / found
    folder.mkdir(parents=True, exist_ok=True)
    ready = sorted(folder.glob("audio.*"))
    if ready:
        return ready[0]
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(folder / "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as client:
            client.extract_info(url, download=True)
    except Exception as exc:
        raise ValueError(f"YouTube download failed: {exc}") from exc
    ready = sorted(folder.glob("audio.*"))
    if not ready:
        raise ValueError("YouTube did not return an audio file.")
    return ready[0]


def _host(hostname: str) -> str:
    if hostname.startswith("www."):
        return hostname[4:]
    return hostname


def _youtube(url: str) -> bool:
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and _host(parsed.hostname or "") in _HOSTS
