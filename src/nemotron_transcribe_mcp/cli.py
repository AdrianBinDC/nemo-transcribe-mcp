import logging
import sys
from pathlib import Path

from mcp.server import MCPServer

from nemotron_transcribe_mcp.config import load_config
from nemotron_transcribe_mcp.transcribe import IncompleteTranscript, transcribe_file

log = logging.getLogger("nemotron_transcribe_mcp")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )
    config = load_config()
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
            "The file must fit in one request."
        )
    )
    def transcribe(path: str) -> str:
        try:
            return transcribe_file(Path(path).expanduser(), config=load_config())
        except IncompleteTranscript as exc:
            raise RuntimeError(f"transcript was cut off: {exc.text}") from exc

    log.info("endpoint %s", config.base_url)
    server.run(transport="stdio")
