import argparse
import base64
import json
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import quote

import httpx

from nemo_transcribe_mcp.config import _environment


def list_model_ids(base_url: str) -> list[str]:
    response = httpx.get(base_url.rstrip("/") + "/models", timeout=1.5)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [
        item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def ffmpeg_version() -> str | None:
    if shutil.which("ffmpeg") is None:
        return None
    proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
    lines = proc.stdout.splitlines()
    return lines[0] if lines else "ffmpeg"


def cursor_install_line() -> str:
    raw = json.dumps(
        {"command": "uvx", "args": ["nemo-transcribe-mcp"]},
        separators=(",", ":"),
    ).encode()
    encoded = quote(base64.b64encode(raw).decode("ascii"), safe="")
    return f"cursor://anysphere.cursor-deeplink/mcp/install?name=nemo-transcribe&config={encoded}"


def _transcribe_model(model_ids: list[str]) -> str | None:
    for model_id in model_ids:
        if "transcribe" in model_id.lower():
            return model_id
    return None


def _find(list_models: Callable[[str], list[str]], urls: tuple[str, ...]) -> tuple[str, str] | None:
    for base_url in urls:
        try:
            model_id = _transcribe_model(list_models(base_url))
        except (OSError, httpx.HTTPError, ValueError, KeyError, TypeError):
            continue
        if model_id:
            return base_url, model_id
    return None


def _write_config(path: Path, base_url: str, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'base_url = "{base_url}"\nmodel = "{model}"\n', encoding="utf-8")


def _saved(path: Path) -> tuple[str, str] | None:
    if not path.is_file():
        return None
    loaded = tomllib.loads(path.read_text(encoding="utf-8"))
    base_url = loaded.get("base_url")
    model = loaded.get("model")
    if isinstance(base_url, str) and base_url.strip():
        return base_url.strip().rstrip("/"), model.strip() if isinstance(model, str) else ""
    return None


def _urls(explicit: str | None, environ: Mapping[str, str] | None) -> tuple[str, ...]:
    if explicit:
        return (explicit.rstrip("/"),)
    configured = _environment(environ).get("NEMOTRON_BASE_URL", "").strip().rstrip("/")
    if configured:
        return (configured,)
    return ()


def run_init(
    argv: list[str],
    *,
    config_path: Path,
    list_models: Callable[[str], list[str]] = list_model_ids,
    ffmpeg_version: Callable[[], str | None] = ffmpeg_version,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="nemo-transcribe-mcp init")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    urls = _urls(args.base_url, environ)
    found = _find(list_models, urls)
    if found is None:
        print("No Transcribe server found.")
        print("Point at one: nemo-transcribe-mcp init --base-url http://HOST:PORT/v1")
        return 1
    base_url, model = found
    _write_config(config_path, base_url, model)
    print(f"Found {model} at {base_url}")
    print(f"Saved {config_path}")
    version = ffmpeg_version()
    if version is None:
        print("ffmpeg is missing. Install ffmpeg, then run init again.")
        return 1
    print(f"ffmpeg ok {version}")
    print("Word timestamps are off.")
    print("Paste this into Cursor:")
    print(cursor_install_line())
    return 0


def run_doctor(
    *,
    config_path: Path,
    list_models: Callable[[str], list[str]] = list_model_ids,
    ffmpeg_version: Callable[[], str | None] = ffmpeg_version,
    base_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    saved = _saved(config_path)
    if base_url:
        target = base_url.rstrip("/")
    elif saved:
        target = saved[0]
    else:
        found = _find(list_models, _urls(None, environ))
        if found is None:
            print("endpoint unreachable")
            print("Point at one: nemo-transcribe-mcp init --base-url http://HOST:PORT/v1")
            return 1
        target = found[0]
    problems = 0
    try:
        model_id = _transcribe_model(list_models(target))
    except (OSError, httpx.HTTPError, ValueError, KeyError, TypeError):
        print(f"endpoint unreachable {target}")
        return 1
    if model_id is None:
        print(f"endpoint has no Transcribe model {target}")
        problems += 1
    else:
        print(f"endpoint ok {target}")
        print(f"model ok {model_id}")
    version = ffmpeg_version()
    if version is None:
        print("ffmpeg is missing. Install ffmpeg, then run doctor again.")
        problems += 1
    else:
        print(f"ffmpeg ok {version}")
    print("word timestamps off")
    return 1 if problems else 0
