# Nemotron Transcribe MCP

Local [MCP](https://modelcontextprotocol.io) server that transcribes an audio file by calling an already-running [Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) endpoint. It does not download weights or start the model.

Long files are cut at a pause, about every 55 minutes, and sent one piece at a time. Pass `speakers` from 2 to 8 to repeat 10 seconds at each join. Speaker numbers start over on every piece.

## Languages

The server detects the spoken language, and the source or destination language must be English.

### Leading

State of the art, or close to it, out of the box.

| Language | Flag | ISO 639-1 |
| --- | --- | --- |
| Czech | 🇨🇿 | `cs` |
| Dutch | 🇳🇱 | `nl` |
| English | 🇺🇸 | `en` |
| French | 🇫🇷 | `fr` |
| German | 🇩🇪 | `de` |
| Hindi | 🇮🇳 | `hi` |
| Italian | 🇮🇹 | `it` |
| Korean | 🇰🇷 | `ko` |
| Polish | 🇵🇱 | `pl` |
| Portuguese | 🇧🇷 | `pt` |
| Russian | 🇷🇺 | `ru` |
| Slovak | 🇸🇰 | `sk` |
| Spanish | 🇪🇸 | `es` |

### Production-ready

Production-level quality.

| Language | Flag | ISO 639-1 |
| --- | --- | --- |
| Bulgarian | 🇧🇬 | `bg` |
| Croatian | 🇭🇷 | `hr` |
| Danish | 🇩🇰 | `da` |
| Estonian | 🇪🇪 | `et` |
| Finnish | 🇫🇮 | `fi` |
| Hungarian | 🇭🇺 | `hu` |
| Japanese | 🇯🇵 | `ja` |
| Latvian | 🇱🇻 | `lv` |
| Lithuanian | 🇱🇹 | `lt` |
| Romanian | 🇷🇴 | `ro` |
| Slovenian | 🇸🇮 | `sl` |
| Swedish | 🇸🇪 | `sv` |
| Thai | 🇹🇭 | `th` |
| Turkish | 🇹🇷 | `tr` |
| Ukrainian | 🇺🇦 | `uk` |

### Supported

Supported, and fine-tuning can help.

| Language | Flag | ISO 639-1 |
| --- | --- | --- |
| Arabic | 🇪🇬 | `ar` |
| Bengali | 🇧🇩 | `bn` |
| Chinese | 🇨🇳 | `zh` |
| Greek | 🇬🇷 | `el` |
| Hebrew | 🇮🇱 | `he` |
| Indonesian | 🇮🇩 | `id` |
| Maltese | 🇲🇹 | `mt` |
| Marathi | 🇮🇳 | `mr` |
| Norwegian | 🇳🇴 | `no` |
| Tagalog | 🇵🇭 | `tl` |
| Tamil | 🇮🇳 | `ta` |
| Telugu | 🇮🇳 | `te` |
| Vietnamese | 🇻🇳 | `vi` |

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

With no arguments this is an MCP server on stdio. Log lines go to stderr. `NEMOTRON_LOG_LEVEL` in `.env` sets how much is written. The default format is one JSON object per line, so `docker logs` stays readable. Set `NEMOTRON_LOG_FORMAT=text` for a plain line in a terminal.

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
