from importlib.resources import files

import yaml


def _table() -> dict:
    raw = files("nemo_transcribe_mcp").joinpath("prompts.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise TypeError("prompts.yaml is not a table")
    return loaded


def prompts_version() -> str:
    return str(_table()["prompts_version"])


def system_for(style: str) -> str:
    for template in _table()["templates"]:
        if template["id"] == style:
            return str(template.get("system") or "")
    raise KeyError(f"{style} prompt is missing from prompts.yaml")


def prompt_for(style: str) -> str:
    for template in _table()["templates"]:
        if template["id"] == style:
            return str(template["user"])
    raise KeyError(f"{style} prompt is missing from prompts.yaml")


def verbatim_prompt() -> str:
    return prompt_for("verbatim")


def translation_prompt(target_language: str) -> str:
    return prompt_for("translation").format(target_language=target_language)


def language_id_prompt(label_set: str = "fleurs", candidates: list[str] | None = None) -> str:
    if candidates:
        labels = ", ".join(item.strip() for item in candidates if item.strip())
        if not labels:
            raise ValueError("candidates must include a label")
        return prompt_for("language_id_candidates").format(labels=labels)
    if label_set == "fleurs":
        return prompt_for("language_id_fleurs")
    if label_set == "voxlingua":
        return prompt_for("language_id_voxlingua")
    raise ValueError("label_set must be fleurs or voxlingua")
