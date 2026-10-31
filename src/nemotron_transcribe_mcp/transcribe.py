import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

from nemotron_transcribe_mcp.audio import detect_silences, normalize_to_pcm, slice_pcm
from nemotron_transcribe_mcp.config import Config
from nemotron_transcribe_mcp.logsetup import get_logger
from nemotron_transcribe_mcp.plan import Piece, plan_pieces, stitch_transcripts
from nemotron_transcribe_mcp.prompts import verbatim_prompt
from nemotron_transcribe_mcp.tokens import output_token_cap

HARD_MAX_S = 3300
log = get_logger("transcribe")


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


def _pieces_for(path: Path, duration_s: float, hard_max_s: float, speakers: bool) -> list[Piece]:
    if duration_s <= hard_max_s:
        return [Piece(0.0, duration_s)]
    silences = detect_silences(path)
    pieces = plan_pieces(duration_s, silences, hard_max_s=hard_max_s, speakers=speakers)
    if any(piece.follows_hard_cut for piece in pieces):
        silences = detect_silences(path, noise_db=-30, min_silence_s=0.3)
        pieces = plan_pieces(duration_s, silences, hard_max_s=hard_max_s, speakers=speakers)
    return pieces


def _post_transcript(
    client: httpx.Client,
    config: Config,
    url: str,
    headers: dict[str, str],
    wav: bytes,
    duration_s: float,
) -> str:
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
    response = client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("model content was not text")
    if choice.get("finish_reason") == "length":
        raise IncompleteTranscript(content)
    return content


def _transcribe_span(
    client: httpx.Client,
    config: Config,
    url: str,
    headers: dict[str, str],
    source: Path,
    work: Path,
    start_s: float,
    end_s: float,
    label: str,
    *,
    can_split: bool,
) -> str:
    dest = work / f"{label}.wav"
    slice_pcm(source, dest, start_s, end_s)
    audio_s = _duration_s(dest)
    started = time.monotonic()
    try:
        text = _post_transcript(
            client,
            config,
            url,
            headers,
            dest.read_bytes(),
            audio_s,
        )
    except IncompleteTranscript:
        log.warning("request_cut_off", audio_s=round(audio_s, 3))
        if not can_split:
            raise
        mid_s = start_s + (end_s - start_s) / 2
        left = _transcribe_span(
            client, config, url, headers, source, work, start_s, mid_s, f"{label}a", can_split=False
        )
        right = _transcribe_span(
            client, config, url, headers, source, work, mid_s, end_s, f"{label}b", can_split=False
        )
        return stitch_transcripts([left, right])
    log.info(
        "request_done",
        audio_s=round(audio_s, 3),
        duration_ms=round((time.monotonic() - started) * 1000, 1),
    )
    return text


def transcribe_file(
    path: Path,
    *,
    config: Config,
    transport: httpx.BaseTransport | None = None,
    hard_max_s: float = HARD_MAX_S,
    speakers: bool = False,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    url = f"{config.base_url}/chat/completions"
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        wav_path = work / "norm.wav"
        normalize_to_pcm(path, wav_path)
        duration_s = _duration_s(wav_path)
        pieces = _pieces_for(wav_path, duration_s, hard_max_s, speakers)
        log.info(
            "pieces_planned",
            pieces=len(pieces),
            audio_s=round(duration_s, 3),
            hard_cuts=sum(piece.follows_hard_cut for piece in pieces),
        )
        with httpx.Client(transport=transport, timeout=1800.0) as client:
            parts = [
                _transcribe_span(
                    client,
                    config,
                    url,
                    headers,
                    wav_path,
                    work,
                    piece.start_s,
                    piece.end_s,
                    f"piece{index}",
                    can_split=True,
                )
                for index, piece in enumerate(pieces)
            ]
    return stitch_transcripts(parts)
