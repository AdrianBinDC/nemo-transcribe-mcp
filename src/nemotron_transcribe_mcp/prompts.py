from importlib.resources import files

import yaml


def _table() -> dict:
    raw = files("nemotron_transcribe_mcp").joinpath("prompts.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise TypeError("prompts.yaml is not a table")
    return loaded


def prompts_version() -> str:
    return str(_table()["prompts_version"])


def verbatim_prompt() -> str:
    for template in _table()["templates"]:
        if template["id"] == "verbatim":
            return str(template["user"])
    raise KeyError("verbatim prompt is missing from prompts.yaml")
