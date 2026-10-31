import pytest
from httpx import ASGITransport, AsyncClient
from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse

from nemotron_transcribe_mcp.cli import _bearer_app


@pytest.mark.anyio
async def test_healthz_is_open_and_mcp_needs_bearer() -> None:
    server = MCPServer(name="test")

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    app = _bearer_app(server.streamable_http_app(streamable_http_path="/mcp"), "secret")
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        denied = await client.post("/mcp")
        assert denied.status_code == 401
