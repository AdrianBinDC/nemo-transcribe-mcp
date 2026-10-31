import json
import subprocess
from pathlib import Path

from nemotron_transcribe_mcp.audio import normalize_to_pcm


def test_normalize_makes_16khz_mono_pcm(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.3",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    dest = tmp_path / "norm.wav"
    normalize_to_pcm(source, dest)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "json",
            str(dest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "pcm_s16le"
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1
