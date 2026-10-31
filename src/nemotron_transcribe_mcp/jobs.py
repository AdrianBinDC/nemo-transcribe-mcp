import hashlib
import json
import secrets
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from nemotron_transcribe_mcp.plan import stitch_transcripts


@dataclass(frozen=True)
class Job:
    id: str
    cached: bool
    state: str


@dataclass(frozen=True)
class Chunk:
    text: str


_ACTIVE = ("queued", "running", "done")


def _idem_key(path: Path, settings: dict[str, object]) -> str:
    data = path.read_bytes()
    window = 1024 * 1024
    first = data[:window]
    last = data[-window:] if data else b""
    source = hashlib.sha256(f"{len(data)}:".encode() + first + b":" + last).hexdigest()
    body = json.dumps({"settings": settings, "source": source}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


class JobStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._done: dict[str, threading.Event] = {}
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                idem_key TEXT NOT NULL,
                state TEXT NOT NULL,
                source_path TEXT NOT NULL,
                result_text TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS chunks (
                job_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                start_s REAL NOT NULL,
                end_s REAL NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY (job_id, idx)
            );
            """
        )
        self._conn.execute("UPDATE jobs SET state = 'queued' WHERE state = 'running'")
        self._conn.commit()

    def submit(self, audio: Path, settings: dict[str, object]) -> Job:
        key = _idem_key(audio, settings)
        with self._lock:
            row = self._conn.execute(
                "SELECT id, state FROM jobs WHERE idem_key = ? AND state IN (?, ?, ?)",
                (key, *_ACTIVE),
            ).fetchone()
            if row is not None:
                return Job(row["id"], cached=row["state"] == "done", state=row["state"])
            job_id = "j_" + secrets.token_hex(16)
            self._conn.execute(
                "INSERT INTO jobs (id, idem_key, state, source_path) VALUES (?, ?, 'queued', ?)",
                (job_id, key, str(audio)),
            )
            self._conn.commit()
            self._done.setdefault(job_id, threading.Event())
            return Job(job_id, cached=False, state="queued")

    def get(self, job_id: str) -> Job:
        with self._lock:
            row = self._conn.execute("SELECT id, state FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("missing")
        return Job(row["id"], cached=row["state"] == "done", state=row["state"])

    def finish(self, job_id: str, text: str) -> None:
        self._set_result(job_id, "done", text)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE jobs SET state = 'running' WHERE id = ?", (job_id,))
            self._conn.commit()

    def mark_cancelled(self, job_id: str, text: str) -> None:
        self._set_result(job_id, "cancelled", text)

    def fail(self, job_id: str, text: str) -> None:
        self._set_result(job_id, "failed", text)

    def save_chunk(self, job_id: str, idx: int, start_s: float, end_s: float, text: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO chunks (job_id, idx, start_s, end_s, text)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id, idx) DO UPDATE SET text = excluded.text
                """,
                (job_id, idx, start_s, end_s, text),
            )
            self._conn.commit()

    def chunks(self, job_id: str) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT text FROM chunks WHERE job_id = ? ORDER BY idx",
                (job_id,),
            ).fetchall()
        return [Chunk(row["text"]) for row in rows]

    def result_text(self, job_id: str) -> str:
        with self._lock:
            row = self._conn.execute("SELECT result_text FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("missing")
        return row["result_text"] or ""

    def request_cancel(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
            self._conn.commit()
            event = self._events.get(job_id)
        if event is not None:
            event.set()

    def bind_cancel(self, job_id: str) -> threading.Event:
        with self._lock:
            event = threading.Event()
            row = self._conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is not None and row["cancel_requested"]:
                event.set()
            self._events[job_id] = event
            return event

    def unbind_cancel(self, job_id: str) -> None:
        with self._lock:
            self._events.pop(job_id, None)

    def wait_until_settled(self, job_id: str, seconds: float) -> bool:
        if self.get(job_id).state in {"done", "cancelled", "failed"}:
            return True
        with self._lock:
            event = self._done.setdefault(job_id, threading.Event())
        if self.get(job_id).state in {"done", "cancelled", "failed"}:
            return True
        return event.wait(seconds)

    def list_jobs(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, state, source_path FROM jobs ORDER BY rowid DESC"
            ).fetchall()
        return [
            {
                "job_id": row["id"],
                "state": row["state"],
                "source": {"kind": "file", "path": row["source_path"]},
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _set_result(self, job_id: str, state: str, text: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET state = ?, result_text = ? WHERE id = ?",
                (state, text, job_id),
            )
            self._conn.commit()
            event = self._done.get(job_id)
        if event is not None:
            event.set()


def _message(state: str, detail: str) -> str:
    if state == "queued":
        return (
            "Waiting for the current file to finish. Tell the user it is waiting. "
            "Call get_job_status with this job_id later; do not poll in a loop."
        )
    if state == "done":
        return "Transcription is done. Call get_job_result for the words."
    if state == "failed":
        return detail
    if state == "cancelled":
        return "Cancelled. Call get_job_result for any pieces that finished."
    return (
        "Still transcribing. Tell the user it is running. "
        "Call get_job_status with this job_id later; do not poll in a loop."
    )


def _joined(store: JobStore, job_id: str) -> str:
    return stitch_transcripts(chunk.text for chunk in store.chunks(job_id))


def run_job(
    store: JobStore,
    job_id: str,
    pieces: list[tuple[float, float]],
    piece,
    work: Path,
) -> str:
    cancelled = store.bind_cancel(job_id)
    try:
        store.mark_running(job_id)
        for index, (start_s, end_s) in enumerate(pieces):
            if cancelled.is_set():
                text = _joined(store, job_id)
                store.mark_cancelled(job_id, text)
                return text
            already = store.chunks(job_id)
            if index < len(already):
                continue
            try:
                text = piece(start_s, end_s, cancelled, work)
            except Exception:
                if cancelled.is_set():
                    kept = _joined(store, job_id)
                    store.mark_cancelled(job_id, kept)
                    return kept
                raise
            if cancelled.is_set():
                kept = _joined(store, job_id)
                store.mark_cancelled(job_id, kept)
                return kept
            store.save_chunk(job_id, index, start_s, end_s, text)
        text = _joined(store, job_id)
        store.finish(job_id, text)
        return text
    finally:
        store.unbind_cancel(job_id)


class Jobs:
    def __init__(self, path: Path, *, plan, transcribe_piece) -> None:
        self._store = JobStore(path)
        self._plan = plan
        self._transcribe_piece = transcribe_piece
        self._threads: dict[str, threading.Thread] = {}
        self._threads_lock = threading.Lock()
        self._slot = threading.Lock()

    def transcribe(self, audio: Path, settings: dict[str, object], *, wait_seconds: float):
        job = self._store.submit(audio, settings)
        if job.cached:
            return {
                "job_id": job.id,
                "cached": True,
                "state": "done",
                "message_for_agent": "This file is already transcribed.",
            }
        self._start(job.id, audio, settings)
        if wait_seconds and self._store.wait_until_settled(job.id, wait_seconds):
            loaded = self._store.get(job.id)
            if loaded.state == "done":
                return self._store.result_text(job.id)
        loaded = self._store.get(job.id)
        return {
            "job_id": job.id,
            "state": loaded.state,
            "cached": False,
            "message_for_agent": _message(loaded.state, self._store.result_text(job.id)),
        }

    def status(self, job_id: str, wait_seconds: float = 0) -> dict[str, object]:
        self._store.get(job_id)
        if wait_seconds:
            self._store.wait_until_settled(job_id, wait_seconds)
        loaded = self._store.get(job_id)
        return {
            "job_id": loaded.id,
            "state": loaded.state,
            "progress": {"chunks_done": len(self._store.chunks(job_id))},
            "message_for_agent": _message(loaded.state, self._store.result_text(job_id)),
        }

    def cancel(self, job_id: str) -> dict[str, object]:
        self._store.get(job_id)
        loaded = self._store.get(job_id)
        if loaded.state in {"done", "cancelled", "failed"}:
            return self.status(job_id)
        self._store.request_cancel(job_id)
        with self._threads_lock:
            thread = self._threads.get(job_id)
            alive = thread is not None and thread.is_alive()
        if not alive:
            self._store.mark_cancelled(job_id, _joined(self._store, job_id))
        return self.status(job_id)

    def result(self, job_id: str, offset: int | None = None, limit: int | None = None) -> str:
        text = self._store.result_text(job_id)
        if offset is None:
            return text
        end = None if limit is None else offset + limit
        return text[offset:end]

    def list(self) -> list[dict[str, object]]:
        return self._store.list_jobs()

    def close(self) -> None:
        with self._threads_lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=5)

    def _start(self, job_id: str, audio: Path, settings: dict[str, object]) -> None:
        with self._threads_lock:
            running = self._threads.get(job_id)
            if running is not None and running.is_alive():
                return

            def work() -> None:
                with self._slot:
                    if self._store.get(job_id).state in {"cancelled", "done", "failed"}:
                        return
                    folder = Path(tempfile.mkdtemp(prefix="nemotron-job-"))
                    try:
                        spans = self._plan(audio, settings, folder)

                        def one(start_s: float, end_s: float, cancelled: threading.Event, work_path: Path) -> str:
                            return self._transcribe_piece(
                                audio, settings, start_s, end_s, cancelled, work_path
                            )

                        run_job(self._store, job_id, spans, one, folder)
                    except Exception as exc:
                        if self._store.get(job_id).state not in {"done", "cancelled", "failed"}:
                            self._store.fail(job_id, str(exc))

            thread = threading.Thread(target=work, daemon=True)
            self._threads[job_id] = thread
            thread.start()
