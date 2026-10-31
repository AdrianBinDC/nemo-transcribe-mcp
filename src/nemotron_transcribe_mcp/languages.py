_TRANSLATION = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "uk": "Ukrainian",
}

_BY_KEY = {code: name for code, name in _TRANSLATION.items()}
_BY_KEY.update({name.lower(): name for name in _TRANSLATION.values()})
_BY_KEY.update({"en": "English", "english": "English"})


def translation_pair(target_language: str, source_language: str = "English") -> tuple[str, str]:
    source_name = _BY_KEY.get(source_language.strip().lower())
    target_name = _BY_KEY.get(target_language.strip().lower())
    if source_name is None:
        raise ValueError(f"{source_language} is not a supported translation language")
    if target_name is None:
        supported = ", ".join(sorted(_TRANSLATION.values()))
        raise ValueError(f"{target_language} is not a supported translation language. Supported: {supported}")
    if source_name != "English" and target_name != "English":
        raise ValueError("one side must be English")
    if source_name == "English" and target_name == "English":
        raise ValueError("one side must be a language other than English")
    return source_name, target_name
