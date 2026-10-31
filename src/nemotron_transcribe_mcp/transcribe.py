import base64
import json
import subprocess
import tempfile
from pathlib import Path

import httpx

from nemotron_transcribe_mcp.audio import normalize_to_pcm
from nemotron_transcribe_mcp.config import Config
from nemotron_transcribe_mcp.prompts import verbatim_prompt
from nemotron_transcribe_mcp.tokens import output_token_cap


class IncompleteTranscript(Exception):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__("transcript was cut off")


def _duration_s(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(proc.stdout)["format"]["duration"])


def transcribe_file(
    path: Path,
    *,
    config: Config,
    transport: httpx.BaseTransport | None = None,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    with tempfile.TemporaryDirectory() as folder:
        wav_path = Path(folder) / "norm.wav"
        normalize_to_pcm(path, wav_path)
        duration_s = _duration_s(wav_path)
        wav = wav_path.read_bytes()

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": verbatim_prompt()},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": output_token_cap(duration_s),
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    url = f"{config.base_url}/chat/completions"
    with httpx.Client(transport=transport, timeout=120.0) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()
    choice = body["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("model content was not text")
    if choice.get("finish_reason") == "length":
        raise IncompleteTranscript(content)
    return content
