import logging
import sys

from mcp.server import MCPServer

from nemotron_transcribe_mcp.config import load_config

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
            "Does not start the model."
        ),
    )
    log.info("endpoint %s", config.base_url)
    server.run(transport="stdio")
