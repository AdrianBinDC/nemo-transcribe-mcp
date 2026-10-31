from pathlib import Path

import pytest

from nemotron_transcribe_mcp.cli import build_server
from nemotron_transcribe_mcp.config import Config
from nemotron_transcribe_mcp.jobs import Jobs
from nemotron_transcribe_mcp.youtube import resolve_audio, video_id

_SHORT = "https://www.youtube.com/shorts/abcdefghijk"


def test_video_id_reads_a_short_a_watch_link_and_a_short_link() -> None:
    assert video_id("https://www.youtube.com/shorts/abcdefghijk") == "abcdefghijk"
    assert video_id("https://www.youtube.com/watch?v=abcdefghijk") == "abcdefghijk"
    assert video_id("https://youtu.be/abcdefghijk") == "abcdefghijk"
    assert video_id("https://m.youtube.com/watch?v=abcdefghijk&t=3") == "abcdefghijk"


def test_a_playlist_link_has_no_video() -> None:
    assert video_id("https://www.youtube.com/playlist?list=PL123") is None


def test_other_links_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only a YouTube link works"):
        resolve_audio("https://example.com/talk.mp3", lambda url: tmp_path / "audio.m4a")


def test_a_youtube_link_without_a_video_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no video"):
        resolve_audio(
            "https://www.youtube.com/playlist?list=PL123", lambda url: tmp_path / "audio.m4a"
        )


def test_a_file_path_is_left_alone(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    assert resolve_audio(str(audio), lambda url: tmp_path / "nope") == audio


@pytest.mark.anyio
async def test_transcribe_saves_a_youtube_link_before_the_job(tmp_path: Path) -> None:
    saved = tmp_path / "audio.m4a"
    seen: list[Path] = []

    def fetch(url: str) -> Path:
        assert url == _SHORT
        saved.write_bytes(b"audio")
        return saved

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(path)
        return "hello"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(
        jobs,
        Config(base_url="http://127.0.0.1:8010/v1", model="m", api_key=""),
        fetch_youtube=fetch,
    )
    text = (
        (await server.call_tool("transcribe", {"path": _SHORT, "wait_seconds": 2})).content[0].text
    )
    assert text == "hello"
    assert seen == [saved]
    jobs.close()


@pytest.mark.anyio
async def test_transcribe_tells_the_caller_when_the_link_is_not_youtube(tmp_path: Path) -> None:
    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=lambda *args: "nope",
    )
    server = build_server(jobs, Config(base_url="http://127.0.0.1:8010/v1", model="m", api_key=""))
    with pytest.raises(Exception) as raised:
        await server.call_tool(
            "transcribe", {"path": "https://example.com/talk.mp3", "wait_seconds": 1}
        )
    assert (
        str(raised.value.__cause__)
        == "Only a YouTube link works. Pass a file path for other audio."
    )
    jobs.close()
