import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, find_dotenv
from platformdirs import user_config_dir

from nemotron_transcribe_mcp.languages import language_code


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"


LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    api_key: str
    log_level: str = "INFO"
    log_format: str = "json"
    source_language: str = ""
    target_language: str = "en"


def default_config_path() -> Path:
    return Path(user_config_dir("nemotron-transcribe-mcp", ensure_exists=False)) / "config.toml"


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is not None:
        return environ
    merged = dict(os.environ)
    dotenv_path = find_dotenv(usecwd=True)
    if not dotenv_path:
        return merged
    for key, value in dotenv_values(dotenv_path).items():
        if value and not merged.get(key, "").strip():
            merged[key] = value
    return merged


def load_config(
    environ: Mapping[str, str] | None = None,
    path: Path | None = None,
) -> Config:
    env = _environment(environ)
    if path is None:
        path = default_config_path()
    saved: dict[str, object] = {}
    if path.is_file():
        loaded = tomllib.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            saved = loaded

    def pick(env_name: str, file_key: str, default: str) -> str:
        from_env = env.get(env_name, "").strip()
        if from_env:
            return from_env
        from_file = saved.get(file_key, "")
        if isinstance(from_file, str) and from_file.strip():
            return from_file.strip()
        return default

    base_url = pick("NEMOTRON_BASE_URL", "base_url", "").rstrip("/")
    if not base_url:
        raise ValueError(
            "NEMOTRON_BASE_URL is not set. Copy .env.example to .env and set the address of your Transcribe server."
        )
    log_level = pick("NEMOTRON_LOG_LEVEL", "log_level", "INFO").upper()
    if log_level not in LOG_LEVELS:
        raise ValueError("NEMOTRON_LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR.")
    log_format = pick("NEMOTRON_LOG_FORMAT", "log_format", "json").lower()
    if log_format not in {"json", "text"}:
        raise ValueError("NEMOTRON_LOG_FORMAT must be json or text.")
    raw_source = pick("NEMOTRON_SOURCE_LANGUAGE", "source_language", "")
    raw_target = pick("NEMOTRON_TARGET_LANGUAGE", "target_language", "en")
    try:
        source_language = language_code(raw_source) if raw_source else ""
    except ValueError:
        raise ValueError("NEMOTRON_SOURCE_LANGUAGE must be a code from .env.example, such as en or pl.") from None
    try:
        target_language = language_code(raw_target)
    except ValueError:
        raise ValueError("NEMOTRON_TARGET_LANGUAGE must be a code from .env.example, such as en or th.") from None
    return Config(
        base_url=base_url,
        model=pick("NEMOTRON_MODEL", "model", DEFAULT_MODEL),
        api_key=pick("NEMOTRON_API_KEY", "api_key", ""),
        log_level=log_level,
        log_format=log_format,
        source_language=source_language,
        target_language=target_language,
    )
