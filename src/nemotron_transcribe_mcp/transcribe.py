import base64
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx

from nemotron_transcribe_mcp.audio import detect_silences, normalize_to_pcm, slice_pcm
from nemotron_transcribe_mcp.config import Config
from nemotron_transcribe_mcp.logsetup import get_logger
from nemotron_transcribe_mcp.plan import (
    Piece,
    plan_pieces,
    speech_window,
    stitch_transcripts,
)
from nemotron_transcribe_mcp.prompts import prompt_for
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


def answer_text(message: dict[str, object], *, show_reasoning: bool) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        raise TypeError("model content was not text")
    reasoning = message.get("reasoning_content")
    if show_reasoning and isinstance(reasoning, str) and reasoning.strip():
        return f"{reasoning.strip()}\n\n{content}"
    return content


def _post_transcript(
    client: httpx.Client,
    config: Config,
    url: str,
    headers: dict[str, str],
    wav: bytes,
    duration_s: float,
    style: str,
    prompt: str | None = None,
    system: str | None = None,
    reasoning: bool = False,
    show_reasoning: bool = False,
) -> str:
    user_text = prompt if prompt is not None else prompt_for(style)
    audio_part = {
        "type": "audio_url",
        "audio_url": {"url": "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")},
    }
    messages: list[dict[str, object]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": [{"type": "text", "text": user_text}, audio_part]})
    if reasoning:
        payload: dict[str, object] = {
            "model": config.model,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 20480,
            "seed": 0,
            "top_k": -1,
            "thinking_token_budget": 17408,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 16384},
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
    else:
        payload = {
            "model": config.model,
            "messages": messages,
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
    message = choice["message"]
    if not isinstance(message, dict):
        raise TypeError("model content was not text")
    text = answer_text(message, show_reasoning=show_reasoning)
    if choice.get("finish_reason") == "length":
        raise IncompleteTranscript(text)
    return text


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
    style: str,
    prompt: str | None = None,
    system: str | None = None,
    reasoning: bool = False,
    show_reasoning: bool = False,
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
            style,
            prompt,
            system,
            reasoning,
            show_reasoning,
        )
    except IncompleteTranscript:
        log.warning("request_cut_off", audio_s=round(audio_s, 3))
        if not can_split or reasoning:
            raise
        mid_s = start_s + (end_s - start_s) / 2
        left = _transcribe_span(
            client,
            config,
            url,
            headers,
            source,
            work,
            start_s,
            mid_s,
            f"{label}a",
            can_split=False,
            style=style,
            prompt=prompt,
        )
        right = _transcribe_span(
            client,
            config,
            url,
            headers,
            source,
            work,
            mid_s,
            end_s,
            f"{label}b",
            can_split=False,
            style=style,
            prompt=prompt,
        )
        return stitch_transcripts([left, right])
    log.info(
        "request_done",
        audio_s=round(audio_s, 3),
        duration_ms=round((time.monotonic() - started) * 1000, 1),
    )
    return text


def prepare_language_clip(
    path: Path, work: Path, *, length_s: float = 10
) -> list[tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    if length_s <= 0:
        raise ValueError("clip_seconds must be greater than 0")
    wav_path = work / "norm.wav"
    normalize_to_pcm(path, wav_path, max_seconds=120)
    duration_s = _duration_s(wav_path)
    start_s, end_s = speech_window(duration_s, detect_silences(wav_path), length_s=length_s)
    return [(start_s, end_s)]


def prepare_clip(path: Path, work: Path, *, max_seconds: float) -> list[tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    if max_seconds <= 0:
        raise ValueError("clip_seconds must be greater than 0")
    wav_path = work / "norm.wav"
    normalize_to_pcm(path, wav_path, max_seconds=max_seconds)
    return [(0.0, _duration_s(wav_path))]


def prepare_pieces(
    path: Path,
    work: Path,
    *,
    speakers: bool,
    hard_max_s: float = HARD_MAX_S,
) -> list[tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
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
    return [(piece.start_s, piece.end_s) for piece in pieces]


def transcribe_prepared_span(
    source: Path,
    work: Path,
    start_s: float,
    end_s: float,
    *,
    config: Config,
    cancelled: threading.Event,
    style: str = "verbatim",
    prompt: str | None = None,
    system: str | None = None,
    reasoning: bool = False,
    show_reasoning: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> str:
    if cancelled.is_set():
        raise RuntimeError("aborted")
    wav_path = work / "norm.wav"
    if not wav_path.is_file():
        normalize_to_pcm(source, wav_path)
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    url = f"{config.base_url}/chat/completions"
    label = f"span-{start_s:.3f}-{end_s:.3f}"
    with httpx.Client(transport=transport, timeout=1800.0) as client:
        finished = threading.Event()

        def watch() -> None:
            while not finished.is_set():
                if cancelled.wait(0.2):
                    client.close()
                    return

        threading.Thread(target=watch, daemon=True).start()
        try:
            return _transcribe_span(
                client,
                config,
                url,
                headers,
                wav_path,
                work,
                start_s,
                end_s,
                label,
                can_split=not reasoning,
                style=style,
                prompt=prompt,
                system=system,
                reasoning=reasoning,
                show_reasoning=show_reasoning,
            )
        except Exception:
            if cancelled.is_set():
                raise RuntimeError("aborted") from None
            raise
        finally:
            finished.set()


def media_duration_s(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")
    return _duration_s(path)


def transcribe_file(
    path: Path,
    *,
    config: Config,
    transport: httpx.BaseTransport | None = None,
    hard_max_s: float = HARD_MAX_S,
    speakers: bool = False,
    style: str = "verbatim",
    prompt: str | None = None,
) -> str:
    headers: dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    url = f"{config.base_url}/chat/completions"
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        spans = prepare_pieces(path, work, speakers=speakers, hard_max_s=hard_max_s)
        wav_path = work / "norm.wav"
        with httpx.Client(transport=transport, timeout=1800.0) as client:
            parts = [
                _transcribe_span(
                    client,
                    config,
                    url,
                    headers,
                    wav_path,
                    work,
                    start_s,
                    end_s,
                    f"piece{index}",
                    can_split=True,
                    style=style,
                    prompt=prompt,
                )
                for index, (start_s, end_s) in enumerate(spans)
            ]
    return stitch_transcripts(parts)
