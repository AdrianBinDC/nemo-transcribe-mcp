import base64
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nemotron_transcribe_mcp.setupcheck import run_doctor, run_init

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"


def _models(base_url: str) -> list[str]:
    if base_url == "http://127.0.0.1:8000/v1":
        return ["some-other-model"]
    if base_url == "http://127.0.0.1:8010/v1":
        return [MODEL]
    raise OSError(base_url)


def _cursor_config(text: str) -> dict[str, object]:
    line = next(item for item in text.splitlines() if item.startswith("cursor://"))
    encoded = parse_qs(urlparse(line).query)["config"][0]
    return json.loads(base64.b64decode(encoded))


def test_init_saves_the_transcribe_server_and_prints_the_cursor_line(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.toml"

    code = run_init([], config_path=path, list_models=_models, ffmpeg_version=lambda: "ffmpeg version 7")

    out = capsys.readouterr().out
    saved = path.read_text(encoding="utf-8")
    assert code == 0
    assert 'base_url = "http://127.0.0.1:8010/v1"' in saved
    assert MODEL in saved
    assert "off" in out.lower()
    assert _cursor_config(out) == {"command": "uvx", "args": ["nemotron-transcribe-mcp"]}


def test_init_stops_when_no_transcribe_server_answers(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.toml"

    def down(base_url: str) -> list[str]:
        raise OSError(base_url)

    code = run_init([], config_path=path, list_models=down, ffmpeg_version=lambda: "ffmpeg version 7")

    assert code == 1
    assert not path.exists()
    assert "--base-url" in capsys.readouterr().out


def test_init_stops_when_ffmpeg_is_missing(tmp_path: Path, capsys) -> None:
    code = run_init(
        ["--base-url", "http://127.0.0.1:8010/v1"],
        config_path=tmp_path / "config.toml",
        list_models=_models,
        ffmpeg_version=lambda: None,
    )

    assert code == 1
    assert "ffmpeg" in capsys.readouterr().out.lower()


def test_doctor_reports_word_timestamps_off(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'base_url = "http://127.0.0.1:8010/v1"\nmodel = "{MODEL}"\n', encoding="utf-8")

    code = run_doctor(config_path=path, list_models=_models, ffmpeg_version=lambda: "ffmpeg version 7")

    out = capsys.readouterr().out.lower()
    assert code == 0
    assert "off" in out
    assert "ffmpeg" in out


def test_doctor_fails_when_the_endpoint_is_down(tmp_path: Path, capsys) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f'base_url = "http://127.0.0.1:8010/v1"\nmodel = "{MODEL}"\n', encoding="utf-8")

    def down(base_url: str) -> list[str]:
        raise OSError(base_url)

    code = run_doctor(config_path=path, list_models=down, ffmpeg_version=lambda: "ffmpeg version 7")

    assert code == 1
    assert "endpoint" in capsys.readouterr().out.lower()
