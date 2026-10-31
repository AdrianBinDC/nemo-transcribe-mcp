import json
import threading
import time
import wave
from pathlib import Path

import pytest

from nemo_transcribe_mcp.cli import build_server
from nemo_transcribe_mcp.config import Config
from nemo_transcribe_mcp.jobs import Jobs


def _config() -> Config:
    return Config(base_url="http://127.0.0.1:8010/v1", model="m", api_key="")


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "clip.bin"
    path.write_bytes(b"x" * 64)
    return path


def _text(result) -> str:
    return result.content[0].text


@pytest.mark.anyio
async def test_clean_style_does_not_reuse_a_verbatim_job(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    seen: list[str] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(str(settings["style"]))
        return str(settings["style"])

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())

    verbatim = await server.call_tool("transcribe", {"path": str(audio), "wait_seconds": 2})
    clean = await server.call_tool(
        "transcribe",
        {"path": str(audio), "style": "clean", "wait_seconds": 2},
    )

    assert _text(verbatim) == "verbatim"
    assert _text(clean) == "clean"
    assert seen == ["verbatim", "clean"]
    jobs.close()


@pytest.mark.anyio
async def test_technical_style_does_not_reuse_another_style(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    seen: list[str] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(str(settings["style"]))
        return str(settings["style"])

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())

    clean = await server.call_tool(
        "transcribe",
        {"path": str(audio), "style": "clean", "wait_seconds": 2},
    )
    technical = await server.call_tool(
        "transcribe",
        {"path": str(audio), "style": "technical", "wait_seconds": 2},
    )

    assert _text(clean) == "clean"
    assert _text(technical) == "technical"
    assert seen == ["clean", "technical"]
    jobs.close()


@pytest.mark.anyio
async def test_raw_dictation_does_not_reuse_another_style(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    seen: list[str] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(str(settings["style"]))
        return str(settings["style"])

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())

    technical = await server.call_tool(
        "transcribe",
        {"path": str(audio), "style": "technical", "wait_seconds": 2},
    )
    raw = await server.call_tool(
        "transcribe",
        {"path": str(audio), "style": "raw_dictation", "wait_seconds": 2},
    )

    assert _text(technical) == "technical"
    assert _text(raw) == "raw_dictation"
    assert seen == ["technical", "raw_dictation"]
    jobs.close()


@pytest.mark.anyio
async def test_speakers_with_technical_style_is_rejected(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=lambda path, settings, start, end, cancelled, work: "nope",
    )
    server = build_server(jobs, _config())

    with pytest.raises(Exception) as raised:
        await server.call_tool(
            "transcribe",
            {"path": str(audio), "style": "technical", "speakers": 2, "wait_seconds": 2},
        )
    assert str(raised.value.__cause__) == "speakers cannot be combined with style technical"

    jobs.close()


@pytest.mark.anyio
async def test_translate_to_french_does_not_reuse_a_transcript(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    seen: list[dict[str, object]] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(settings)
        return str(settings["style"])

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())

    words = await server.call_tool("transcribe", {"path": str(audio), "wait_seconds": 2})
    french = await server.call_tool(
        "translate",
        {"path": str(audio), "target_language": "fr", "wait_seconds": 2},
    )

    assert _text(words) == "verbatim"
    assert _text(french) == "translation"
    assert seen[1]["target_language"] == "French"
    assert seen[1]["prompt"] == "Please translate the given speech to French."
    jobs.close()


@pytest.mark.anyio
async def test_translate_uses_the_configured_language_when_the_target_is_empty(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    seen: list[dict[str, object]] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(settings)
        return "bonjour"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(
        jobs,
        Config(base_url="http://127.0.0.1:8010/v1", model="m", api_key="", target_language="fr"),
    )

    await server.call_tool("translate", {"path": str(audio), "wait_seconds": 2})

    assert seen[0]["target_language"] == "French"
    jobs.close()


@pytest.mark.anyio
async def test_a_named_language_overrides_the_configured_one(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    seen: list[dict[str, object]] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(settings)
        return "hello"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(
        jobs,
        Config(
            base_url="http://127.0.0.1:8010/v1",
            model="m",
            api_key="",
            source_language="de",
            target_language="en",
        ),
    )

    await server.call_tool(
        "translate",
        {"path": str(audio), "source_language": "pl", "target_language": "en", "wait_seconds": 2},
    )

    assert seen[0]["source_language"] == "Polish"
    assert seen[0]["detect_source"] is False
    jobs.close()


@pytest.mark.anyio
async def test_the_file_overrides_the_configured_source_when_the_call_does_not(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    seen: list[dict[str, object]] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(settings)
        return "hello"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(
        jobs,
        Config(
            base_url="http://127.0.0.1:8010/v1",
            model="m",
            api_key="",
            source_language="de",
            target_language="en",
        ),
    )

    await server.call_tool("translate", {"path": str(audio), "wait_seconds": 2})

    assert seen[0]["source_language"] == ""
    assert seen[0]["detect_source"] is True
    assert seen[0]["fallback_source"] == "de"
    jobs.close()


@pytest.mark.anyio
async def test_translate_call_names_the_spoken_language_and_the_written_language(
    tmp_path: Path,
) -> None:
    audio = _audio(tmp_path)
    seen: list[dict[str, object]] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(settings)
        return "hello"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())

    await server.call_tool(
        "translate",
        {"path": str(audio), "source_language": "pl", "target_language": "en", "wait_seconds": 2},
    )

    assert seen[0]["source_language"] == "Polish"
    assert seen[0]["target_language"] == "English"
    assert seen[0]["detect_source"] is False
    jobs.close()


@pytest.mark.anyio
async def test_translate_rejects_two_non_english_languages(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=lambda path, settings, start, end, cancelled, work: "nope",
    )
    server = build_server(jobs, _config())

    with pytest.raises(Exception) as raised:
        await server.call_tool(
            "translate",
            {
                "path": str(audio),
                "source_language": "pl",
                "target_language": "th",
                "wait_seconds": 2,
            },
        )
    assert str(raised.value.__cause__) == (
        "Polish to Thai is not supported. The source or destination must be English."
    )
    jobs.close()


@pytest.mark.anyio
async def test_ask_about_audio_requires_a_question(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=lambda path, settings, start, end, cancelled, work: "nope",
    )
    server = build_server(jobs, _config())
    with pytest.raises(Exception) as raised:
        await server.call_tool("ask_about_audio", {"path": str(audio), "question": "  "})
    assert str(raised.value.__cause__) == "question is required"
    jobs.close()


@pytest.mark.anyio
async def test_ask_about_audio_sends_the_question(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    with wave.open(str(source), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    seen: list[dict[str, object]] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(settings)
        return "FedEx"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(float(settings["start_s"]), float(settings["end_s"]))],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())
    answer = await server.call_tool(
        "ask_about_audio",
        {"path": str(source), "question": "Who is named?", "wait_seconds": 2},
    )
    assert _text(answer) == "FedEx"
    assert seen[0]["style"] == "audio_qa"
    assert seen[0]["prompt"] == "Who is named?"
    assert seen[0]["start_s"] == 0.0
    jobs.close()


@pytest.mark.anyio
async def test_identify_language_sends_the_fleurs_labels(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    seen: list[dict[str, object]] = []

    def piece(path, settings, start, end, cancelled, work) -> str:
        seen.append(settings)
        return "en_us"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())

    result = await server.call_tool(
        "identify_language",
        {"path": str(audio), "wait_seconds": 2},
    )

    assert _text(result) == "en_us"
    assert seen[0]["style"] == "language_id"
    assert seen[0]["clip_seconds"] == 10
    assert "en_us" in str(seen[0]["prompt"])
    assert "fr_fr" in str(seen[0]["prompt"])
    jobs.close()


@pytest.mark.anyio
async def test_a_fast_file_returns_the_words(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=lambda path, settings, start, end, cancelled, work: "hello",
    )
    server = build_server(jobs, _config())

    result = await server.call_tool("transcribe", {"path": str(audio), "wait_seconds": 2})

    assert _text(result) == "hello"
    jobs.close()


@pytest.mark.anyio
async def test_a_slow_file_returns_a_job_the_agent_can_finish(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    def piece(path, settings, start, end, cancelled, work) -> str:
        time.sleep(0.2)
        return "hello"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())

    started = json.loads(
        _text(await server.call_tool("transcribe", {"path": str(audio), "wait_seconds": 0}))
    )
    assert started["job_id"].startswith("j_")
    assert "get_job_status" in started["message_for_agent"]

    status = json.loads(
        _text(
            await server.call_tool(
                "get_job_status", {"job_id": started["job_id"], "wait_seconds": 2}
            )
        )
    )
    assert status["state"] == "done"
    assert _text(await server.call_tool("get_job_result", {"job_id": started["job_id"]})) == "hello"
    listed = json.loads(_text(await server.call_tool("list_jobs", {})))
    assert listed[0]["job_id"] == started["job_id"]
    jobs.close()


@pytest.mark.anyio
async def test_cancel_job_stops_the_piece_that_is_running(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    entered = threading.Event()

    def piece(path, settings, start, end, cancelled, work) -> str:
        entered.set()
        if cancelled.wait(2):
            raise RuntimeError("aborted")
        return "nope"

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    server = build_server(jobs, _config())
    started = json.loads(
        _text(await server.call_tool("transcribe", {"path": str(audio), "wait_seconds": 0}))
    )
    assert entered.wait(1)

    await server.call_tool("cancel_job", {"job_id": started["job_id"]})
    status = json.loads(
        _text(
            await server.call_tool(
                "get_job_status", {"job_id": started["job_id"], "wait_seconds": 2}
            )
        )
    )
    assert status["state"] == "cancelled"
    jobs.close()


@pytest.mark.anyio
async def test_list_languages_returns_the_support_levels(tmp_path: Path) -> None:
    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=lambda *args: "",
    )
    server = build_server(jobs, _config())
    rows = json.loads(_text(await server.call_tool("list_languages", {})))
    french = next(row for row in rows if row["name"] == "French")
    assert french["level"] == "leading"
    assert french["translate"] == "yes"
    jobs.close()
