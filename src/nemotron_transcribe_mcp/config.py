import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_dir


DEFAULT_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    api_key: str


def default_config_path() -> Path:
    return Path(user_config_dir("nemotron-transcribe-mcp", ensure_exists=False)) / "config.toml"


def load_config(
    environ: Mapping[str, str] | None = None,
    path: Path | None = None,
) -> Config:
    env = os.environ if environ is None else environ
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

    return Config(
        base_url=pick("NEMOTRON_BASE_URL", "base_url", DEFAULT_BASE_URL).rstrip("/"),
        model=pick("NEMOTRON_MODEL", "model", DEFAULT_MODEL),
        api_key=pick("NEMOTRON_API_KEY", "api_key", ""),
    )
