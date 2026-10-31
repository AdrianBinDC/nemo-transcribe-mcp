import json
import threading
import time
from pathlib import Path

import pytest

from nemotron_transcribe_mcp.cli import build_server
from nemotron_transcribe_mcp.config import Config
from nemotron_transcribe_mcp.jobs import Jobs


def _config() -> Config:
    return Config(base_url="http://127.0.0.1:8010/v1", model="m", api_key="")


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "clip.bin"
    path.write_bytes(b"x" * 64)
    return path


def _text(result) -> str:
    return result.content[0].text


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

    started = json.loads(_text(await server.call_tool("transcribe", {"path": str(audio), "wait_seconds": 0})))
    assert started["job_id"].startswith("j_")
    assert "get_job_status" in started["message_for_agent"]

    status = json.loads(
        _text(await server.call_tool("get_job_status", {"job_id": started["job_id"], "wait_seconds": 2}))
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
    started = json.loads(_text(await server.call_tool("transcribe", {"path": str(audio), "wait_seconds": 0})))
    assert entered.wait(1)

    await server.call_tool("cancel_job", {"job_id": started["job_id"]})
    status = json.loads(
        _text(await server.call_tool("get_job_status", {"job_id": started["job_id"], "wait_seconds": 2}))
    )
    assert status["state"] == "cancelled"
    jobs.close()
