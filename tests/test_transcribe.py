import base64
import io
import json
import subprocess
import wave
from pathlib import Path

import httpx

import pytest

from nemotron_transcribe_mcp.config import Config
from nemotron_transcribe_mcp.plan import Silence
from nemotron_transcribe_mcp.transcribe import IncompleteTranscript, transcribe_file

SPARK = Config(
    base_url="http://127.0.0.1:8010/v1",
    model="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
    api_key="test-key",
)


def _one_second_wav(path: Path) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


def test_short_file_returns_the_model_transcript(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    _one_second_wav(source)
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello from the booth."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    text = transcribe_file(source, config=SPARK, transport=httpx.MockTransport(handler))

    assert text == "Hello from the booth."
    request = seen["request"]
    assert str(request.url) == "http://127.0.0.1:8010/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == SPARK.model
    assert body["temperature"] == 0
    assert body["top_p"] == 1.0
    assert body["seed"] == 0
    assert body["max_tokens"] == 65011
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["skip_special_tokens"] is False
    assert body["spaces_between_special_tokens"] is False
    assert "mm_processor_kwargs" not in body
    content = body["messages"][0]["content"]
    assert content[0]["text"] == (
        "Audio transcription. Language: as spoken. "
        "Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization. "
        "Output: transcript only."
    )
    url = content[1]["audio_url"]["url"]
    encoded = url.removeprefix("data:audio/wav;base64,")
    assert encoded != url
    with wave.open(io.BytesIO(base64.b64decode(encoded))) as sent:
        assert sent.getnchannels() == 1
        assert sent.getframerate() == 16000
        assert sent.getsampwidth() == 2
        assert sent.getnframes() == 16000


def test_clean_style_asks_the_model_to_drop_fillers(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    _one_second_wav(source)
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello from the booth."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    transcribe_file(source, config=SPARK, transport=httpx.MockTransport(handler), style="clean")

    body = json.loads(seen["request"].content)
    assert body["messages"][0]["content"][0]["text"] == (
        "Audio transcription. Language: as spoken. "
        "Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. "
        "Formatting: punctuation and capitalization. "
        "Output: transcript only."
    )


def test_technical_style_asks_the_model_to_keep_exact_strings(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    _one_second_wav(source)
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "mail at example.com"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    transcribe_file(source, config=SPARK, transport=httpx.MockTransport(handler), style="technical")

    body = json.loads(seen["request"].content)
    assert body["messages"][0]["content"][0]["text"] == (
        "Transcribe as a clean written transcript. "
        "Preserve exact values and punctuation/separators in emails, URLs, file paths, commands, and identifiers. "
        "Use conventional written forms."
    )


def test_raw_dictation_asks_for_spoken_numbers_and_no_punctuation(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    _one_second_wav(source)
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "twenty nine"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    transcribe_file(source, config=SPARK, transport=httpx.MockTransport(handler), style="raw_dictation")

    body = json.loads(seen["request"].content)
    assert body["messages"][0]["content"][0]["text"] == (
        "Transcribe word for word without punctuation. "
        "Write numbers as spoken words and keep individually spoken letters separate."
    )


def test_length_finish_means_the_transcript_was_cut_off(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    _one_second_wav(source)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "partial words"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    with pytest.raises(IncompleteTranscript, match="cut off") as raised:
        transcribe_file(source, config=SPARK, transport=httpx.MockTransport(handler))
    assert raised.value.text == "partial words"


def _gap_wav(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-filter_complex",
            "[0][1][2]concat=n=3:v=0:a=1",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_long_file_is_sent_one_piece_at_a_time(tmp_path: Path) -> None:
    source = tmp_path / "show.wav"
    _gap_wav(source)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        text = "left side" if len(calls) == 1 else "right side"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    text = transcribe_file(
        source,
        config=SPARK,
        transport=httpx.MockTransport(handler),
        hard_max_s=5,
    )

    assert text == "left side\nright side"
    assert len(calls) == 2
    second = json.loads(calls[1].content)
    encoded = second["messages"][0]["content"][1]["audio_url"]["url"].removeprefix(
        "data:audio/wav;base64,"
    )
    with wave.open(io.BytesIO(base64.b64decode(encoded))) as sent:
        # The pause is at about 4.5 s, so the second piece is the tail, not the whole file.
        assert 3 < sent.getnframes() / sent.getframerate() < 4


def test_looser_silence_pass_runs_when_the_first_finds_no_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "show.wav"
    _gap_wav(source)
    seen: list[tuple[float, float]] = []

    def fake_detect(path: Path, *, noise_db: float = -35, min_silence_s: float = 0.5) -> list[Silence]:
        del path
        seen.append((noise_db, min_silence_s))
        if len(seen) == 1:
            return []
        return [Silence(4.0, 5.0)]

    monkeypatch.setattr("nemotron_transcribe_mcp.transcribe.detect_silences", fake_detect)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "words"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    transcribe_file(source, config=SPARK, transport=httpx.MockTransport(handler), hard_max_s=5)

    assert seen == [(-35, 0.5), (-30, 0.3)]


def test_speakers_repeat_ten_seconds_at_the_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "talk.wav"
    with wave.open(str(source), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 20)
    monkeypatch.setattr(
        "nemotron_transcribe_mcp.transcribe.detect_silences",
        lambda path, **kwargs: [Silence(12.0, 13.0)],
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "words"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    transcribe_file(
        source,
        config=SPARK,
        transport=httpx.MockTransport(handler),
        hard_max_s=14,
        speakers=True,
    )

    encoded = json.loads(calls[1].content)["messages"][0]["content"][1]["audio_url"]["url"]
    encoded = encoded.removeprefix("data:audio/wav;base64,")
    with wave.open(io.BytesIO(base64.b64decode(encoded))) as sent:
        # Cut at 12.5 s, then start 10 s earlier: 2.5 s through 20 s.
        assert 17 < sent.getnframes() / sent.getframerate() < 18


def test_a_cut_off_piece_is_split_and_sent_again(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    _one_second_wav(source)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            text = "partial words"
            finish = "length"
        else:
            text = "left half" if len(calls) == 2 else "right half"
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish,
                    }
                ]
            },
        )

    text = transcribe_file(source, config=SPARK, transport=httpx.MockTransport(handler))

    assert text == "left half\nright half"
    assert len(calls) == 3


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.wav"
    with pytest.raises(FileNotFoundError, match="nope.wav"):
        transcribe_file(missing, config=SPARK, transport=httpx.MockTransport(lambda request: httpx.Response(500)))
