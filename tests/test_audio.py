import json
import subprocess
from pathlib import Path

from nemotron_transcribe_mcp.audio import normalize_to_pcm, parse_silences
from nemotron_transcribe_mcp.plan import Silence


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


def test_parse_silences_reads_ffmpeg_pairs() -> None:
    log = (
        "silence_start: 1.0\n"
        "silence_end: 3.5 | silence_duration: 2.5\n"
        "[something else]\n"
        "silence_start: 10\n"
        "silence_end: 10.25 | silence_duration: 0.25\n"
    )
    assert parse_silences(log) == [
        Silence(1.0, 3.5),
        Silence(10, 10.25),
    ]
