from pathlib import Path

import pytest

from nemotron_transcribe_mcp.config import load_config


def test_environment_overrides_saved_config(tmp_path: Path) -> None:
    saved = tmp_path / "config.toml"
    saved.write_text(
        'base_url = "http://127.0.0.1:8000/v1"\n'
        'model = "saved-model"\n'
        'api_key = "saved-key"\n'
    )

    config = load_config(
        {
            "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1",
            "NEMOTRON_MODEL": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
            "NEMOTRON_API_KEY": "env-key",
        },
        path=saved,
    )

    assert config.base_url == "http://127.0.0.1:8010/v1"
    assert config.model == "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    assert config.api_key == "env-key"


def test_saved_config_is_used_when_environment_is_empty(tmp_path: Path) -> None:
    saved = tmp_path / "config.toml"
    saved.write_text(
        'base_url = "http://127.0.0.1:8010/v1"\n'
        'model = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"\n'
        'api_key = "file-key"\n'
    )

    config = load_config({}, path=saved)

    assert config.base_url == "http://127.0.0.1:8010/v1"
    assert config.model == "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    assert config.api_key == "file-key"


def test_builtin_default_points_at_the_spark_transcribe_port(tmp_path: Path) -> None:
    config = load_config({}, path=tmp_path / "missing.toml")

    assert config.base_url == "http://127.0.0.1:8010/v1"
    assert config.model == "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    assert config.api_key == ""


def test_base_url_trailing_slash_is_removed() -> None:
    config = load_config(
        {
            "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1/",
            "NEMOTRON_MODEL": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
        }
    )

    assert config.base_url == "http://127.0.0.1:8010/v1"


def test_unset_path_reads_the_user_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    folder = tmp_path / "nemotron-transcribe-mcp"
    folder.mkdir()
    (folder / "config.toml").write_text(
        'base_url = "http://10.0.0.8:8010/v1"\n',
        encoding="utf-8",
    )

    config = load_config({})

    assert config.base_url == "http://10.0.0.8:8010/v1"
