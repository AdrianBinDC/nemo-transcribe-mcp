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


def prompt_for(style: str) -> str:
    for template in _table()["templates"]:
        if template["id"] == style:
            return str(template["user"])
    raise KeyError(f"{style} prompt is missing from prompts.yaml")


def verbatim_prompt() -> str:
    return prompt_for("verbatim")
