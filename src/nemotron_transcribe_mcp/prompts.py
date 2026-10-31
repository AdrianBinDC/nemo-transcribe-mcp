from importlib.resources import files

import yaml


def verbatim_prompt() -> str:
    raw = files("nemotron_transcribe_mcp").joinpath("prompts.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    for template in loaded["templates"]:
        if template["id"] == "verbatim":
            return str(template["user"])
    raise KeyError("verbatim prompt is missing from prompts.yaml")
