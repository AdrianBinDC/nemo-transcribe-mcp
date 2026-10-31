import subprocess
from pathlib import Path

from nemo_transcribe_mcp.plan import Silence


def normalize_to_pcm(source: Path, dest: Path, *, max_seconds: float | None = None) -> None:
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-threads",
        "2",
    ]
    if max_seconds is not None:
        command.extend(["-t", f"{max_seconds:.3f}"])
    command.extend(
        [
            "-i",
            str(source),
            "-vn",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )
    subprocess.run(command, check=True, capture_output=True)


def detect_silences(
    path: Path,
    *,
    noise_db: float = -35,
    min_silence_s: float = 0.5,
) -> list[Silence]:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_silences(proc.stderr)


def slice_pcm(source: Path, dest: Path, start_s: float, end_s: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(source),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def parse_silences(log: str) -> list[Silence]:
    pending: list[float] = []
    found: list[Silence] = []
    for line in log.splitlines():
        if "silence_start:" in line:
            pending.append(float(line.split("silence_start:", 1)[1].split()[0]))
        elif "silence_end:" in line and pending:
            end_s = float(line.split("silence_end:", 1)[1].split()[0])
            found.append(Silence(pending.pop(0), end_s))
    return found
