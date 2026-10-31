# Nemotron Transcribe MCP

Local [MCP](https://modelcontextprotocol.io) server that transcribes an audio file by calling an already-running [Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) endpoint. It does not download weights or start the model.

Long files are cut at a pause, about every 55 minutes, and sent one piece at a time. Pass `speakers` from 2 to 8 to repeat 10 seconds at each join. Speaker numbers start over on every piece.

This server writes a transcript. For timed subtitles (SRT or VTT), use another model. This checkpoint returns no word times.

## Languages

The server hears the language in the file.

| From | To |
| --- | --- |
| Supported language | Same supported language |
| Non-English Supported | English |
| English | Non-English Supported |

### Leading

State of the art, or close to it, out of the box.

| Language | Flag | ISO 639-1 | Translate |
| --- | --- | --- | --- |
| Czech | 🇨🇿 | `cs` | Yes |
| Dutch | 🇳🇱 | `nl` | Yes |
| English | 🇺🇸 | `en` | Yes |
| French | 🇫🇷 | `fr` | Yes |
| German | 🇩🇪 | `de` | Yes |
| Hindi | 🇮🇳 | `hi` | Yes |
| Italian | 🇮🇹 | `it` | Yes |
| Korean | 🇰🇷 | `ko` | Yes |
| Polish | 🇵🇱 | `pl` | Yes |
| Portuguese | 🇧🇷 | `pt` | Yes |
| Russian | 🇷🇺 | `ru` | Yes |
| Slovak | 🇸🇰 | `sk` | Yes |
| Spanish | 🇪🇸 | `es` | Yes |

### Production-ready

Production-level quality.

| Language | Flag | ISO 639-1 | Translate |
| --- | --- | --- | --- |
| Bulgarian | 🇧🇬 | `bg` | Yes |
| Croatian | 🇭🇷 | `hr` | Yes |
| Danish | 🇩🇰 | `da` | Yes |
| Estonian | 🇪🇪 | `et` | Yes |
| Finnish | 🇫🇮 | `fi` | Yes |
| Hungarian | 🇭🇺 | `hu` | Yes |
| Japanese | 🇯🇵 | `ja` | Yes |
| Latvian | 🇱🇻 | `lv` | Yes |
| Lithuanian | 🇱🇹 | `lt` | Yes |
| Romanian | 🇷🇴 | `ro` | Yes |
| Slovenian | 🇸🇮 | `sl` | Yes |
| Swedish | 🇸🇪 | `sv` | Yes |
| Thai | 🇹🇭 | `th` | Yes |
| Turkish | 🇹🇷 | `tr` | No |
| Ukrainian | 🇺🇦 | `uk` | Yes |

### Supported

Supported, and fine-tuning can help.

| Language | Flag | ISO 639-1 | Translate |
| --- | --- | --- | --- |
| Arabic | 🇪🇬 | `ar` | Yes |
| Bengali | 🇧🇩 | `bn` | No |
| Chinese | 🇨🇳 | `zh` | Yes |
| Greek | 🇬🇷 | `el` | Yes |
| Hebrew | 🇮🇱 | `he` | Yes |
| Indonesian | 🇮🇩 | `id` | No |
| Maltese | 🇲🇹 | `mt` | Yes |
| Marathi | 🇮🇳 | `mr` | No |
| Norwegian | 🇳🇴 | `no` | No |
| Tagalog | 🇵🇭 | `tl` | No |
| Tamil | 🇮🇳 | `ta` | No |
| Telugu | 🇮🇳 | `te` | No |
| Vietnamese | 🇻🇳 | `vi` | No |

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
