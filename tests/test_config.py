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


def test_log_level_comes_from_the_environment(tmp_path: Path) -> None:
    config = load_config(
        {
            "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1",
            "NEMOTRON_LOG_LEVEL": "debug",
            "NEMOTRON_LOG_FORMAT": "text",
        },
        path=tmp_path / "missing.toml",
    )

    assert config.log_level == "DEBUG"
    assert config.log_format == "text"


def test_source_language_is_blank_and_the_written_language_defaults_to_english(tmp_path: Path) -> None:
    config = load_config(
        {"NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1"},
        path=tmp_path / "missing.toml",
    )

    assert config.source_language == ""
    assert config.target_language == "en"


def test_language_accepts_a_code_or_a_name(tmp_path: Path) -> None:
    config = load_config(
        {
            "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1",
            "NEMOTRON_SOURCE_LANGUAGE": "Polish",
            "NEMOTRON_TARGET_LANGUAGE": "en",
        },
        path=tmp_path / "missing.toml",
    )

    assert config.source_language == "pl"
    assert config.target_language == "en"


def test_unknown_language_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NEMOTRON_TARGET_LANGUAGE"):
        load_config(
            {
                "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1",
                "NEMOTRON_TARGET_LANGUAGE": "Klingon",
            },
            path=tmp_path / "missing.toml",
        )


def test_unknown_log_level_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NEMOTRON_LOG_LEVEL"):
        load_config(
            {
                "NEMOTRON_BASE_URL": "http://127.0.0.1:8010/v1",
                "NEMOTRON_LOG_LEVEL": "loud",
            },
            path=tmp_path / "missing.toml",
        )


def test_missing_base_url_names_the_env_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"NEMOTRON_BASE_URL"):
        load_config({}, path=tmp_path / "missing.toml")


def test_dotenv_file_supplies_the_base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NEMOTRON_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        'NEMOTRON_BASE_URL=http://10.1.2.3:8010/v1\n',
        encoding="utf-8",
    )

    config = load_config(path=tmp_path / "missing.toml")

    assert config.base_url == "http://10.1.2.3:8010/v1"


def test_process_environment_overrides_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEMOTRON_BASE_URL", "http://127.0.0.1:8010/v1")
    (tmp_path / ".env").write_text(
        'NEMOTRON_BASE_URL=http://127.0.0.1:9000/v1\n',
        encoding="utf-8",
    )

    config = load_config(path=tmp_path / "missing.toml")

    assert config.base_url == "http://127.0.0.1:8010/v1"


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
