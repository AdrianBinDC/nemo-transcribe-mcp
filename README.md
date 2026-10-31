# Nemotron Transcribe MCP

Local [MCP](https://modelcontextprotocol.io) server that transcribes an audio file by calling an already-running [Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) endpoint. It does not download weights or start the model.

Long files are cut at a pause, about every 55 minutes, and sent one piece at a time. Pass `speakers` from 2 to 8 to repeat 10 seconds at each join. Speaker numbers start over on every piece.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` on your `PATH`

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set `NEMOTRON_BASE_URL` to your server. The value includes `/v1`, for example `http://127.0.0.1:8000/v1`. `.env` stays on your machine.

Settings are read in this order: environment variables, then `.env` in the working directory, then `~/.config/nemotron-transcribe-mcp/config.toml`. If the base URL is still missing, the server exits and tells you to set it.

## Run

```bash
uv run nemotron-transcribe-mcp
```

With no arguments this is an MCP server on stdio. Log lines go to stderr.

A Cursor project can launch it from this directory:

```json
{
  "mcpServers": {
    "nemotron-transcribe": {
      "command": "uv",
      "args": ["run", "nemotron-transcribe-mcp"]
    }
  }
}
```

Ask the client to transcribe a file path. The tool is `transcribe`.

## Tests

```bash
uv run pytest
```

The tests do not call a model. They do need `ffmpeg` and `ffprobe`.
