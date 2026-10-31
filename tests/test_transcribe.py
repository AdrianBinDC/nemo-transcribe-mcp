import base64
import io
import json
import wave
from pathlib import Path

import httpx

import pytest

from nemotron_transcribe_mcp.config import Config
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


def test_missing_file_names_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.wav"
    with pytest.raises(FileNotFoundError, match="nope.wav"):
        transcribe_file(missing, config=SPARK, transport=httpx.MockTransport(lambda request: httpx.Response(500)))
