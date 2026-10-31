import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.anyio
async def test_no_arguments_serves_mcp_and_logs_the_endpoint_on_stderr(tmp_path: Path) -> None:
    err_path = tmp_path / "stderr.txt"
    script = Path(sys.executable).parent / "nemotron-transcribe-mcp"
    with err_path.open("w", encoding="utf-8") as errlog:
        params = StdioServerParameters(
            command=str(script),
            args=[],
            env={
                "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1",
                "NEMOTRON_LOG_FORMAT": "json",
                "XDG_CONFIG_HOME": str(tmp_path / "empty-config"),
                "XDG_DATA_HOME": str(tmp_path / "data"),
            },
        )
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            result = await session.initialize()

    assert result.server_info.name == "nemotron-transcribe"
    stderr = err_path.read_text(encoding="utf-8")
    started = next(
        json.loads(line)
        for line in stderr.splitlines()
        if line.startswith("{") and '"server_started"' in line
    )
    assert started["host"] == "127.0.0.1"


@pytest.mark.anyio
async def test_server_offers_a_transcribe_tool(tmp_path: Path) -> None:
    err_path = tmp_path / "stderr.txt"
    script = Path(sys.executable).parent / "nemotron-transcribe-mcp"
    with err_path.open("w", encoding="utf-8") as errlog:
        params = StdioServerParameters(
            command=str(script),
            args=[],
            env={
                "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1",
                "XDG_CONFIG_HOME": str(tmp_path / "empty-config"),
                "XDG_DATA_HOME": str(tmp_path / "data"),
            },
        )
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            listed = await session.list_tools()

    names = {item.name for item in listed.tools}
    assert names >= {
        "transcribe",
        "translate",
        "identify_language",
        "list_languages",
        "ask_about_audio",
        "get_job_status",
        "get_job_result",
        "list_jobs",
        "cancel_job",
    }
    tool = next(item for item in listed.tools if item.name == "transcribe")
    description = (tool.description or "").lower()
    assert "audio" in description
    assert "pause" in description
    assert "get_job_status" in description
