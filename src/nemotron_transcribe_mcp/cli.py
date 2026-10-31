import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from mcp.server import MCPServer
from platformdirs import user_data_dir

from nemotron_transcribe_mcp.config import Config, default_config_path, load_config
from nemotron_transcribe_mcp.jobs import Jobs
from nemotron_transcribe_mcp.logsetup import configure_logging, get_logger
from nemotron_transcribe_mcp.prompts import prompt_for, prompts_version
from nemotron_transcribe_mcp.setupcheck import run_doctor, run_init
from nemotron_transcribe_mcp.transcribe import (
    IncompleteTranscript,
    prepare_pieces,
    transcribe_prepared_span,
)

log = get_logger("server")


def jobs_db_path() -> Path:
    return Path(user_data_dir("nemotron-transcribe-mcp", ensure_exists=True)) / "jobs.db"


def build_server(jobs: Jobs, config: Config) -> MCPServer:
    server = MCPServer(
        name="nemotron-transcribe",
        instructions=(
            "Talks to an already-running Nemotron 3.5 Transcribe endpoint. "
            "Does not start the model. Use transcribe for one audio file. "
            "A long file returns a job id. Call get_job_status, then get_job_result. "
            "Do not poll in a loop."
        ),
    )

    def settings(speakers: int, style: str) -> dict[str, object]:
        return {
            "speakers": speakers,
            "style": style,
            "model": config.model,
            "prompts_version": prompts_version(),
        }

    @server.tool(
        description=(
            "Speech to text for one audio file. "
            "style verbatim, the default, keeps every word, including fillers. "
            "style clean drops fillers and accidental repetitions and keeps the other words. "
            "style technical keeps emails, URLs, file paths, commands, and IDs exactly. "
            "Do not pass speakers with style technical. "
            "style raw_dictation drops punctuation and writes numbers as spoken words. "
            "A long file is cut at a pause and sent one piece at a time. "
            "Pass speakers from 2 to 8 to repeat 10 seconds at each join. "
            "Speaker numbers are matched across those joins. "
            "Waits wait_seconds (default 25). If it is still running, returns a job id. "
            "Tell the user, then call get_job_status. Do not poll in a loop."
        ),
        structured_output=False,
    )
    def transcribe(path: str, speakers: int = 0, style: str = "verbatim", wait_seconds: float = 25) -> str:
        if speakers != 0 and not 2 <= speakers <= 8:
            raise ValueError("speakers must be from 2 to 8")
        if style not in {"verbatim", "clean", "technical", "raw_dictation"}:
            raise ValueError("style must be verbatim, clean, technical, or raw_dictation")
        if style == "technical" and speakers != 0:
            raise ValueError("speakers cannot be combined with style technical")
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be 0 or more")
        prompt_for(style)
        try:
            outcome = jobs.transcribe(
                Path(path).expanduser(),
                settings(speakers, style),
                wait_seconds=wait_seconds,
            )
        except IncompleteTranscript as exc:
            raise RuntimeError(f"transcript was cut off: {exc.text}") from exc
        if isinstance(outcome, str):
            return outcome
        if outcome.get("state") == "failed":
            raise RuntimeError(str(jobs.result(str(outcome["job_id"]))))
        return json.dumps(outcome)

    @server.tool(
        description=(
            "How far a transcription job has got. "
            "Pass wait_seconds from 0 to 45 to wait for it to finish. "
            "Tell the user the status. Do not poll in a loop."
        ),
        structured_output=False,
    )
    def get_job_status(job_id: str, wait_seconds: float = 0) -> str:
        if not 0 <= wait_seconds <= 45:
            raise ValueError("wait_seconds must be from 0 to 45")
        return json.dumps(jobs.status(job_id, wait_seconds=wait_seconds))

    @server.tool(
        description=(
            "The words from a finished job. "
            "offset and limit return a slice so a long transcript does not flood the chat. "
            "Only text is stored."
        ),
        structured_output=False,
    )
    def get_job_result(job_id: str, format: str = "text", offset: int = 0, limit: int = 8000) -> str:
        if format != "text":
            raise ValueError("only text results are stored")
        return jobs.result(job_id, offset=offset, limit=limit)

    @server.tool(
        description="Jobs on this machine, newest first. Use this to recover a job id.",
        structured_output=False,
    )
    def list_jobs(state: str = "", limit: int = 20) -> str:
        rows = jobs.list()
        if state:
            rows = [row for row in rows if row["state"] == state]
        return json.dumps(rows[:limit])

    @server.tool(
        description=(
            "Stop a transcription job. "
            "Pieces that already finished stay readable through get_job_result."
        ),
        structured_output=False,
    )
    def cancel_job(job_id: str) -> str:
        return json.dumps(jobs.cancel(job_id))

    return server


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "init":
        raise SystemExit(run_init(sys.argv[2:], config_path=default_config_path()))
    if command == "doctor":
        raise SystemExit(run_doctor(config_path=default_config_path()))
    if command:
        print("Use init, doctor, or no arguments for the server.", file=sys.stderr)
        raise SystemExit(2)
    try:
        config = load_config()
    except ValueError as exc:
        configure_logging("INFO")
        get_logger("server").error("config_invalid", message=str(exc))
        raise SystemExit(1) from exc
    configure_logging(config.log_level, fmt=config.log_format, secrets=[config.api_key])
    log.info("server_started", host=urlparse(config.base_url).hostname or "")

    def plan(path: Path, settings: dict[str, object], work: Path) -> list[tuple[float, float]]:
        return prepare_pieces(path, work, speakers=int(settings["speakers"]) >= 2)

    def piece(path, settings, start, end, cancelled, work) -> str:
        try:
            return transcribe_prepared_span(
                path,
                work,
                start,
                end,
                config=config,
                cancelled=cancelled,
                style=str(settings.get("style", "verbatim")),
            )
        except IncompleteTranscript as exc:
            raise RuntimeError(f"transcript was cut off: {exc.text}") from exc

    jobs = Jobs(jobs_db_path(), plan=plan, transcribe_piece=piece)
    build_server(jobs, config).run(transport="stdio")
