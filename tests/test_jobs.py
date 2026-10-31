import threading
import time
from pathlib import Path

import pytest

from nemo_transcribe_mcp.jobs import Jobs, JobStore, run_job


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "clip.bin"
    path.write_bytes(b"x" * 64)
    return path


def _settings() -> dict[str, object]:
    return {"speakers": 0, "model": "m", "prompts_version": "test"}


def test_a_second_request_reuses_the_job_that_is_already_running(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    audio = _audio(tmp_path)
    first = store.submit(audio, _settings())
    second = store.submit(audio, _settings())

    assert first.id.startswith("j_")
    assert second.id == first.id
    assert second.cached is False


def test_a_finished_job_comes_back_as_cached(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    audio = _audio(tmp_path)
    job = store.submit(audio, _settings())
    store.finish(job.id, "hello")

    again = store.submit(audio, _settings())

    assert again.id == job.id
    assert again.cached is True


def test_reopening_the_database_queues_an_interrupted_job_and_keeps_its_piece(
    tmp_path: Path,
) -> None:
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    audio = _audio(tmp_path)
    job = store.submit(audio, _settings())
    store.mark_running(job.id)
    store.save_chunk(job.id, 0, 0.0, 1.0, "hello")
    store.close()

    reopened = JobStore(db)
    loaded = reopened.get(job.id)

    assert loaded.state == "queued"
    assert reopened.chunks(job.id)[0].text == "hello"


def test_resume_does_not_send_a_piece_that_was_already_stored(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    audio = _audio(tmp_path)
    job = store.submit(audio, _settings())
    store.save_chunk(job.id, 0, 0.0, 1.0, "hello")
    seen: list[float] = []

    def piece(start: float, end: float, cancelled: threading.Event, work: Path) -> str:
        seen.append(start)
        return "there"

    text = run_job(store, job.id, [(0.0, 1.0), (1.0, 2.0)], piece, tmp_path)

    assert seen == [1.0]
    assert text == "hello\nthere"
    assert store.get(job.id).state == "done"


def test_cancel_leaves_the_finished_piece_readable(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    audio = _audio(tmp_path)
    job = store.submit(audio, _settings())
    store.save_chunk(job.id, 0, 0.0, 1.0, "kept")
    store.request_cancel(job.id)
    seen: list[float] = []

    def piece(start: float, end: float, cancelled: threading.Event, work: Path) -> str:
        seen.append(start)
        return "nope"

    text = run_job(store, job.id, [(0.0, 1.0), (1.0, 2.0)], piece, tmp_path)

    assert seen == []
    assert text == "kept"
    assert store.get(job.id).state == "cancelled"


def test_cancel_stops_the_piece_that_is_in_flight(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    audio = _audio(tmp_path)
    job = store.submit(audio, _settings())
    entered = threading.Event()

    def piece(start: float, end: float, cancelled: threading.Event, work: Path) -> str:
        entered.set()
        if cancelled.wait(2):
            raise RuntimeError("aborted")
        return "done"

    worker = threading.Thread(
        target=lambda: run_job(store, job.id, [(0.0, 1.0)], piece, tmp_path),
    )
    worker.start()
    assert entered.wait(1)
    store.request_cancel(job.id)
    worker.join(2)

    assert not worker.is_alive()
    assert store.get(job.id).state == "cancelled"


def test_a_fast_transcription_returns_the_words(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    def plan(path: Path, settings: dict[str, object], work: Path) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    def piece(
        path: Path,
        settings: dict[str, object],
        start: float,
        end: float,
        cancelled: threading.Event,
        work: Path,
    ) -> str:
        return "hello"

    jobs = Jobs(tmp_path / "jobs.db", plan=plan, transcribe_piece=piece)
    result = jobs.transcribe(audio, _settings(), wait_seconds=2)
    jobs.close()

    assert result == "hello"


def test_a_slow_transcription_returns_a_job_id(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    hold = threading.Event()

    def plan(path: Path, settings: dict[str, object], work: Path) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    def piece(
        path: Path,
        settings: dict[str, object],
        start: float,
        end: float,
        cancelled: threading.Event,
        work: Path,
    ) -> str:
        hold.wait(2)
        return "later"

    jobs = Jobs(tmp_path / "jobs.db", plan=plan, transcribe_piece=piece)
    result = jobs.transcribe(audio, _settings(), wait_seconds=0)
    hold.set()
    jobs.close()

    assert isinstance(result, dict)
    assert result["job_id"].startswith("j_")
    assert result["state"] in {"queued", "running"}
    assert "get_job_status" in result["message_for_agent"]


def test_status_waits_until_the_job_finishes(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    def plan(path: Path, settings: dict[str, object], work: Path) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    def piece(
        path: Path,
        settings: dict[str, object],
        start: float,
        end: float,
        cancelled: threading.Event,
        work: Path,
    ) -> str:
        time.sleep(0.2)
        return "hello"

    jobs = Jobs(tmp_path / "jobs.db", plan=plan, transcribe_piece=piece)
    started = jobs.transcribe(audio, _settings(), wait_seconds=0)
    assert isinstance(started, dict)
    status = jobs.status(started["job_id"], wait_seconds=2)
    jobs.close()

    assert status["state"] == "done"
    assert status["progress"]["chunks_done"] == 1
    assert jobs.result(started["job_id"]) == "hello"


def test_result_can_return_a_slice(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    audio = _audio(tmp_path)
    job = store.submit(audio, _settings())
    store.finish(job.id, "abcdefghij")
    jobs = Jobs(tmp_path / "jobs.db", plan=lambda *_: [], transcribe_piece=lambda *_: "")

    assert jobs.result(job.id, offset=2, limit=3) == "cde"
    jobs.close()


def test_list_includes_the_job(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=lambda path, settings, start, end, cancelled, work: "hello",
    )
    jobs.transcribe(audio, _settings(), wait_seconds=2)
    listed = jobs.list()
    jobs.close()

    assert listed[0]["state"] == "done"
    assert listed[0]["source"]["kind"] == "file"


def test_a_finished_file_is_returned_with_cached_set(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    def plan(path: Path, settings: dict[str, object], work: Path) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    def piece(
        path: Path,
        settings: dict[str, object],
        start: float,
        end: float,
        cancelled: threading.Event,
        work: Path,
    ) -> str:
        return "hello"

    jobs = Jobs(tmp_path / "jobs.db", plan=plan, transcribe_piece=piece)
    jobs.transcribe(audio, _settings(), wait_seconds=2)
    again = jobs.transcribe(audio, _settings(), wait_seconds=2)
    jobs.close()

    assert isinstance(again, dict)
    assert again["cached"] is True
    assert again["state"] == "done"


def test_a_second_file_waits_until_the_first_finishes(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"a" * 64)
    second.write_bytes(b"b" * 64)
    started = threading.Event()
    release = threading.Event()
    seen: list[str] = []

    def piece(
        path: Path,
        settings: dict[str, object],
        start: float,
        end: float,
        cancelled: threading.Event,
        work: Path,
    ) -> str:
        seen.append(path.name)
        if path.name == "a.bin":
            started.set()
            release.wait(2)
        return path.name

    jobs = Jobs(
        tmp_path / "jobs.db",
        plan=lambda path, settings, work: [(0.0, 1.0)],
        transcribe_piece=piece,
    )
    jobs.transcribe(first, _settings(), wait_seconds=0)
    assert started.wait(1)
    waiting = jobs.transcribe(second, _settings(), wait_seconds=0)
    assert isinstance(waiting, dict)
    assert waiting["state"] == "queued"
    assert "waiting" in waiting["message_for_agent"].lower()
    assert seen == ["a.bin"]

    release.set()
    status = jobs.status(waiting["job_id"], wait_seconds=2)
    jobs.close()

    assert status["state"] == "done"
    assert seen == ["a.bin", "b.bin"]
    assert jobs.result(waiting["job_id"]) == "b.bin"


def test_unknown_job_is_rejected(tmp_path: Path) -> None:
    jobs = Jobs(tmp_path / "jobs.db", plan=lambda *_: [], transcribe_piece=lambda *_: "")
    with pytest.raises(KeyError, match="missing"):
        jobs.status("missing")
    jobs.close()
