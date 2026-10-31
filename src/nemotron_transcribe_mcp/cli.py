from pathlib import Path
from urllib.parse import urlparse

from mcp.server import MCPServer

from nemotron_transcribe_mcp.config import load_config
from nemotron_transcribe_mcp.logsetup import configure_logging, get_logger
from nemotron_transcribe_mcp.transcribe import IncompleteTranscript, transcribe_file

log = get_logger("server")


def main() -> None:
    try:
        config = load_config()
    except ValueError as exc:
        configure_logging("INFO")
        get_logger("server").error("config_invalid", message=str(exc))
        raise SystemExit(1) from exc
    configure_logging(config.log_level, fmt=config.log_format, secrets=[config.api_key])
    log.info("server_started", host=urlparse(config.base_url).hostname or "")
    server = MCPServer(
        name="nemotron-transcribe",
        instructions=(
            "Talks to an already-running Nemotron 3.5 Transcribe endpoint. "
            "Does not start the model. Use transcribe for one audio file."
        ),
    )

    @server.tool(
        description=(
            "Speech to text for one audio file. "
            "Returns the words as spoken, including fillers and repetitions. "
            "A long file is cut at a pause and sent one piece at a time. "
            "Pass speakers from 2 to 8 to repeat 10 seconds at each join. "
            "Speaker numbers start over on every piece."
        )
    )
    def transcribe(path: str, speakers: int = 0) -> str:
        if speakers != 0 and not 2 <= speakers <= 8:
            raise ValueError("speakers must be from 2 to 8")
        try:
            return transcribe_file(
                Path(path).expanduser(),
                config=load_config(),
                speakers=speakers >= 2,
            )
        except IncompleteTranscript as exc:
            raise RuntimeError(f"transcript was cut off: {exc.text}") from exc

    server.run(transport="stdio")
