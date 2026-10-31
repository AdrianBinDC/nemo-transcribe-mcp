import json
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

from mcp.server import MCPServer
from platformdirs import user_data_dir

from nemo_transcribe_mcp.config import Config, default_config_path, load_config
from nemo_transcribe_mcp.jobs import Jobs
from nemo_transcribe_mcp.languages import (
    apply_detected_source,
    language_id_note,
    supported_languages,
    translation_name,
    translation_pair,
)
from nemo_transcribe_mcp.logsetup import configure_logging, get_logger
from nemo_transcribe_mcp.prompts import (
    language_id_prompt,
    prompt_for,
    prompts_version,
    system_for,
    translation_prompt,
)
from nemo_transcribe_mcp.setupcheck import run_doctor, run_init
from nemo_transcribe_mcp.tokens import question_span
from nemo_transcribe_mcp.transcribe import (
    IncompleteTranscript,
    media_duration_s,
    prepare_language_clip,
    prepare_pieces,
    transcribe_prepared_span,
)
from nemo_transcribe_mcp.youtube import download_youtube, resolve_audio

log = get_logger("server")


def jobs_db_path(work_root: str = "") -> Path:
    if work_root:
        root = Path(work_root)
        root.mkdir(parents=True, exist_ok=True)
        return root / "jobs.db"
    return Path(user_data_dir("nemo-transcribe-mcp", ensure_exists=True)) / "jobs.db"


def _bearer_app(app, token: str):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    class BearerGate(BaseHTTPMiddleware):
        def __init__(self, asgi_app, expected: str) -> None:
            super().__init__(asgi_app)
            self.expected = expected

        async def dispatch(self, request: Request, call_next):
            if request.url.path == "/healthz":
                return await call_next(request)
            header = request.headers.get("authorization", "")
            if header != f"Bearer {self.expected}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    return BearerGate(app, token)


def build_server(jobs: Jobs, config: Config, fetch_youtube=None) -> MCPServer:
    fetch = download_youtube if fetch_youtube is None else fetch_youtube

    def audio_file(path: str) -> Path:
        return resolve_audio(path, fetch)

    server = MCPServer(
        name="nemo-transcribe",
        instructions=(
            "Nemotron 3.5 Transcribe is already running. This server does not start the model. "
            "Pass a file path or a YouTube link. "
            "transcribe writes the words in the spoken language. "
            "style verbatim keeps every word. clean drops fillers. "
            "technical keeps emails, URLs, and commands exact. "
            "raw_dictation drops punctuation and writes numbers as spoken words. "
            "Pass speakers from 2 to 8 to mark who spoke. "
            "translate writes the words in another language. "
            "A supported language can stay that language. "
            "A non-English supported language can be written in English. "
            "English can be written in a non-English supported language. "
            "identify_language names the spoken language from a short stretch of speech. "
            "list_languages shows each language and how strong the support is. "
            "ask_about_audio answers one question about English speech. "
            "A long recording is cut at a pause and sent one piece at a time. "
            "If the call is still running, it returns a job id. "
            "Call get_job_status, then get_job_result. Do not poll in a loop. "
            "This server writes a transcript. For SRT or VTT, use another model."
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
            "Speech to text for one audio file or one YouTube link. "
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
    def transcribe(
        path: str, speakers: int = 0, style: str = "verbatim", wait_seconds: float = 25
    ) -> str:
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
                audio_file(path),
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
            "Write the speech in one audio file, or one YouTube link, in another language. "
            "source_language is the language spoken in the file, such as Polish or pl. "
            "target_language is the language the words should be written in, such as English or en. "
            "Pass both when you know them. "
            "A language passed here overrides NEMOTRON_SOURCE_LANGUAGE and NEMOTRON_TARGET_LANGUAGE. "
            "If source_language is empty, the language heard in the file overrides NEMOTRON_SOURCE_LANGUAGE. "
            "Leave target_language empty to use NEMOTRON_TARGET_LANGUAGE. "
            "One of the two languages must be English. "
            "A long file is cut at a pause and sent one piece at a time. "
            "Waits wait_seconds (default 25). If it is still running, returns a job id. "
            "Tell the user, then call get_job_status. Do not poll in a loop."
        ),
        structured_output=False,
    )
    def translate(
        path: str,
        target_language: str = "",
        source_language: str = "",
        wait_seconds: float = 25,
    ) -> str:
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be 0 or more")
        spoken = source_language.strip()
        written = target_language.strip() or config.target_language
        if spoken:
            source_name, target_name = translation_pair(written, spoken)
            detect_source = False
            fallback_source = ""
        else:
            source_name = ""
            target_name = translation_name(written)
            detect_source = True
            fallback_source = config.source_language
        try:
            outcome = jobs.transcribe(
                audio_file(path),
                {
                    "speakers": 0,
                    "style": "translation",
                    "source_language": source_name,
                    "target_language": target_name,
                    "detect_source": detect_source,
                    "fallback_source": fallback_source,
                    "prompt": translation_prompt(target_name),
                    "model": config.model,
                    "prompts_version": prompts_version(),
                },
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
            "Answer a question about one audio file or one YouTube link. "
            "The model was checked on English speech. "
            "A recording longer than the reasoning budget is refused. "
            "Pass start and end, in seconds, for the part you want. "
            "show_reasoning includes the model's reasoning before the answer. "
            "Waits wait_seconds (default 25). If it is still running, returns a job id. "
            "Tell the user, then call get_job_status. Do not poll in a loop."
        ),
        structured_output=False,
    )
    def ask_about_audio(
        path: str,
        question: str,
        start: float | None = None,
        end: float | None = None,
        show_reasoning: bool = False,
        wait_seconds: float = 25,
    ) -> str:
        if not question.strip():
            raise ValueError("question is required")
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be 0 or more")
        audio = audio_file(path)
        start_s, end_s = question_span(media_duration_s(audio), start, end)
        try:
            outcome = jobs.transcribe(
                audio,
                {
                    "speakers": 0,
                    "style": "audio_qa",
                    "prompt": question.strip(),
                    "system": system_for("audio_qa"),
                    "show_reasoning": show_reasoning,
                    "start_s": start_s,
                    "end_s": end_s,
                    "model": config.model,
                    "prompts_version": prompts_version(),
                },
                wait_seconds=wait_seconds,
            )
        except IncompleteTranscript as exc:
            raise RuntimeError(f"answer was cut off: {exc.text}") from exc
        if isinstance(outcome, str):
            return outcome
        if outcome.get("state") == "failed":
            raise RuntimeError(str(jobs.result(str(outcome["job_id"]))))
        return json.dumps(outcome)

    @server.tool(
        description=(
            "Say which language is spoken in one audio file or one YouTube link. "
            "Returns one label, such as en_us. "
            "Listens to about 10 seconds of speech, skipping a pause at the start. "
            "Less reliable for Latvian and Slovenian. "
            "Not a transcript."
        ),
        structured_output=False,
    )
    def identify_language(
        path: str,
        candidates: list[str] | None = None,
        label_set: str = "fleurs",
        clip_seconds: float = 10,
        wait_seconds: float = 25,
    ) -> str:
        if not 1 <= clip_seconds <= 30:
            raise ValueError("clip_seconds must be from 1 to 30")
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be 0 or more")
        prompt = language_id_prompt(label_set, candidates)
        try:
            outcome = jobs.transcribe(
                audio_file(path),
                {
                    "speakers": 0,
                    "style": "language_id",
                    "label_set": "candidates" if candidates else label_set,
                    "clip_seconds": clip_seconds,
                    "prompt": prompt,
                    "model": config.model,
                    "prompts_version": prompts_version(),
                },
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
            "Languages this model can handle, and how strong that support is. "
            "level is leading, production_ready, or supported. "
            "translate is yes when that language can be one side of an English translation. "
            "identify is yes, low, or no. Latvian and Slovenian are low. "
            "Norwegian is listed, but the card only scored transcription. It did not mark it supported."
        ),
        structured_output=False,
    )
    def list_languages() -> str:
        return json.dumps(supported_languages())

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
    def get_job_result(
        job_id: str, format: str = "text", offset: int = 0, limit: int = 8000
    ) -> str:
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
    configure_logging(
        config.log_level,
        fmt=config.log_format,
        secrets=[config.api_key, config.mcp_auth_token],
    )
    log.info("server_started", host=urlparse(config.base_url).hostname or "")

    def prepare(path: Path, settings: dict[str, object], work: Path) -> None:
        if settings.get("style") != "translation" or not settings.get("detect_source"):
            return
        spans = prepare_language_clip(path, work, length_s=10)
        label = transcribe_prepared_span(
            path,
            work,
            spans[0][0],
            spans[0][1],
            config=config,
            cancelled=threading.Event(),
            style="language_id",
            prompt=language_id_prompt(),
        )
        apply_detected_source(settings, label)

    def plan(path: Path, settings: dict[str, object], work: Path) -> list[tuple[float, float]]:
        if settings.get("style") == "language_id":
            return prepare_language_clip(path, work, length_s=float(settings["clip_seconds"]))
        if settings.get("style") == "audio_qa":
            return [(float(settings["start_s"]), float(settings["end_s"]))]
        return prepare_pieces(path, work, speakers=int(settings["speakers"]) >= 2)

    def piece(path, settings, start, end, cancelled, work) -> str:
        try:
            text = transcribe_prepared_span(
                path,
                work,
                start,
                end,
                config=config,
                cancelled=cancelled,
                style=str(settings.get("style", "verbatim")),
                prompt=settings.get("prompt") if isinstance(settings.get("prompt"), str) else None,
                system=settings.get("system") if isinstance(settings.get("system"), str) else None,
                reasoning=settings.get("style") == "audio_qa",
                show_reasoning=bool(settings.get("show_reasoning")),
            )
            if settings.get("style") == "language_id":
                return language_id_note(text)
            return text
        except IncompleteTranscript as exc:
            raise RuntimeError(f"transcript was cut off: {exc.text}") from exc

    jobs = Jobs(
        jobs_db_path(config.work_root),
        plan=plan,
        transcribe_piece=piece,
        prepare=prepare,
    )
    server = build_server(jobs, config)
    if config.transport == "stdio":
        server.run(transport="stdio")
        return
    _run_http(server, config)


def _run_http(server: MCPServer, config: Config) -> None:
    import uvicorn
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    app = _bearer_app(
        server.streamable_http_app(streamable_http_path="/mcp", host=config.mcp_host),
        config.mcp_auth_token,
    )
    log.info("http_listening", host=config.mcp_host, port=config.mcp_port)
    uvicorn.run(app, host=config.mcp_host, port=config.mcp_port, log_level="warning")
