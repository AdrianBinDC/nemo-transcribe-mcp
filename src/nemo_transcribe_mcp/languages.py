from importlib.resources import files

import yaml


def _load_rows() -> tuple[list[dict[str, str]], dict[str, str]]:
    raw = files("nemo_transcribe_mcp").joinpath("languages.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw)
    if not isinstance(loaded, dict):
        raise TypeError("languages.yaml is not a language list")
    rows = loaded.get("languages")
    locales = loaded.get("locales")
    if not isinstance(rows, list) or not isinstance(locales, dict):
        raise TypeError("languages.yaml is not a language list")
    cleaned: list[dict[str, str]] = []
    for row in rows:
        cleaned.append(
            {
                "code": str(row["code"]),
                "name": str(row["name"]),
                "level": str(row["level"]),
                "identify": str(row["identify"]),
                "translate": str(row["translate"]),
            }
        )
    return cleaned, {str(code): str(name) for code, name in locales.items()}


_ROWS, _LOCALES = _load_rows()
_TRANSLATION = {
    row["code"]: row["name"] for row in _ROWS if row["translate"] == "yes" and row["code"] != "en"
}
_BY_KEY = {code: name for code, name in _TRANSLATION.items()}
_BY_KEY.update({name.lower(): name for name in _TRANSLATION.values()})
_BY_KEY.update({"en": "English", "english": "English"})
_CODE_BY_NAME = {name.lower(): code for code, name in _TRANSLATION.items()}
_CODE_BY_NAME["en"] = "en"
_CODE_BY_NAME["english"] = "en"
_CODE_BY_NAME.update(_TRANSLATION)
_LOW_NAMES = {row["name"] for row in _ROWS if row["identify"] == "low"}
_LOW_CONFIDENCE = {row["code"] for row in _ROWS if row["identify"] == "low"}
_LOW_CONFIDENCE.update(code for code, name in _LOCALES.items() if name in _LOW_NAMES)


def supported_languages() -> list[dict[str, str]]:
    return [
        {
            "code": row["code"],
            "name": row["name"],
            "level": row["level"],
            "translate": row["translate"],
            "identify": row["identify"],
        }
        for row in _ROWS
    ]


def language_id_note(text: str) -> str:
    stripped = text.strip()
    label = stripped.split()[0].lower() if stripped else ""
    if label in _LOW_CONFIDENCE:
        return f"{stripped}\nLow confidence for this language."
    return stripped


def language_from_label(label: str) -> str:
    token = label.strip().splitlines()[0].split()[0].lower()
    if token in _LOCALES:
        return _LOCALES[token]
    found = _BY_KEY.get(token)
    if found is None:
        raise ValueError(f"{token} is not a supported translation language")
    return found


def language_code(text: str) -> str:
    found = _CODE_BY_NAME.get(text.strip().lower())
    if found is None:
        raise ValueError(f"{text} is not a supported language code")
    return found


def _unsupported(role: str, language: str, other_role: str, other: str) -> ValueError:
    message = f"{role} {language.strip()} is not supported for translation."
    if other.strip():
        message += f" {other_role} {other.strip()} is set."
    return ValueError(message)


def translation_name(text: str) -> str:
    found = _BY_KEY.get(text.strip().lower())
    if found is None:
        raise _unsupported("destination", text, "", "")
    return found


def apply_detected_source(settings: dict[str, object], label: str) -> None:
    destination = str(settings["target_language"])
    heard = label.strip()
    if not heard:
        fallback = str(settings.get("fallback_source") or "")
        if not fallback:
            raise ValueError("The podcast language could not be detected.")
        source_text = fallback
    else:
        try:
            source_text = language_from_label(heard)
        except ValueError:
            token = heard.splitlines()[0].split()[0]
            raise _unsupported("source", token, "destination", destination) from None
    source_name, target_name = translation_pair(destination, source_text)
    settings["source_language"] = source_name
    settings["target_language"] = target_name


def translation_pair(target_language: str, source_language: str = "English") -> tuple[str, str]:
    source_name = _BY_KEY.get(source_language.strip().lower())
    target_name = _BY_KEY.get(target_language.strip().lower())
    if source_name is None:
        raise _unsupported("source", source_language, "destination", target_language)
    if target_name is None:
        raise _unsupported("destination", target_language, "source", source_language)
    if source_name != "English" and target_name != "English":
        raise ValueError(
            f"{source_name} to {target_name} is not supported. The source or destination must be English."
        )
    if source_name == "English" and target_name == "English":
        raise ValueError(
            "English to English is not supported. One language must be other than English."
        )
    return source_name, target_name
