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
_CODE_BY_NAME = {name.lower(): code for code, name in _TRANSLATION.items()}
_CODE_BY_NAME["en"] = "en"
_CODE_BY_NAME["english"] = "en"
_CODE_BY_NAME.update(_TRANSLATION)

# Card tiers. identify is yes, low, or no.
_SUPPORT = (
    ("ar", "Arabic", "supported", "yes"),
    ("bn", "Bengali", "supported", "no"),
    ("bg", "Bulgarian", "production_ready", "yes"),
    ("zh", "Chinese", "supported", "yes"),
    ("hr", "Croatian", "production_ready", "yes"),
    ("cs", "Czech", "leading", "yes"),
    ("da", "Danish", "production_ready", "yes"),
    ("nl", "Dutch", "leading", "yes"),
    ("en", "English", "leading", "yes"),
    ("et", "Estonian", "production_ready", "yes"),
    ("fi", "Finnish", "production_ready", "yes"),
    ("fr", "French", "leading", "yes"),
    ("de", "German", "leading", "yes"),
    ("el", "Greek", "supported", "yes"),
    ("he", "Hebrew", "supported", "yes"),
    ("hi", "Hindi", "leading", "yes"),
    ("hu", "Hungarian", "production_ready", "yes"),
    ("id", "Indonesian", "supported", "yes"),
    ("it", "Italian", "leading", "yes"),
    ("ja", "Japanese", "production_ready", "yes"),
    ("ko", "Korean", "leading", "yes"),
    ("lv", "Latvian", "production_ready", "low"),
    ("lt", "Lithuanian", "production_ready", "yes"),
    ("mt", "Maltese", "supported", "yes"),
    ("mr", "Marathi", "supported", "no"),
    ("no", "Norwegian", "supported", "no"),
    ("pl", "Polish", "leading", "yes"),
    ("pt", "Portuguese", "leading", "yes"),
    ("ro", "Romanian", "production_ready", "yes"),
    ("ru", "Russian", "leading", "yes"),
    ("sk", "Slovak", "leading", "yes"),
    ("sl", "Slovenian", "production_ready", "low"),
    ("es", "Spanish", "leading", "yes"),
    ("sv", "Swedish", "production_ready", "yes"),
    ("tl", "Tagalog", "supported", "yes"),
    ("ta", "Tamil", "supported", "no"),
    ("te", "Telugu", "supported", "no"),
    ("th", "Thai", "production_ready", "yes"),
    ("tr", "Turkish", "production_ready", "yes"),
    ("uk", "Ukrainian", "production_ready", "yes"),
    ("vi", "Vietnamese", "supported", "yes"),
)


def supported_languages() -> list[dict[str, str]]:
    return [
        {
            "code": code,
            "name": name,
            "level": level,
            "translate": "yes" if code == "en" or code in _TRANSLATION else "no",
            "identify": identify,
        }
        for code, name, level, identify in _SUPPORT
    ]


_LOW_CONFIDENCE = {"lv", "lv_lv", "sl", "sl_si"}


def language_id_note(text: str) -> str:
    stripped = text.strip()
    label = stripped.split()[0].lower() if stripped else ""
    if label in _LOW_CONFIDENCE:
        return f"{stripped}\nLow confidence for this language."
    return stripped


_LOCALE = {
    "ar_eg": "Arabic",
    "bg_bg": "Bulgarian",
    "cmn_hans_cn": "Chinese",
    "hr_hr": "Croatian",
    "cs_cz": "Czech",
    "da_dk": "Danish",
    "nl_nl": "Dutch",
    "en_us": "English",
    "et_ee": "Estonian",
    "fil_ph": "Filipino",
    "fi_fi": "Finnish",
    "fr_fr": "French",
    "de_de": "German",
    "el_gr": "Greek",
    "he_il": "Hebrew",
    "hi_in": "Hindi",
    "hu_hu": "Hungarian",
    "id_id": "Indonesian",
    "it_it": "Italian",
    "ja_jp": "Japanese",
    "ko_kr": "Korean",
    "lv_lv": "Latvian",
    "lt_lt": "Lithuanian",
    "mt_mt": "Maltese",
    "pl_pl": "Polish",
    "pt_br": "Portuguese",
    "ro_ro": "Romanian",
    "ru_ru": "Russian",
    "sk_sk": "Slovak",
    "sl_si": "Slovenian",
    "es_419": "Spanish",
    "sv_se": "Swedish",
    "th_th": "Thai",
    "tr_tr": "Turkish",
    "uk_ua": "Ukrainian",
    "vi_vn": "Vietnamese",
}


def language_from_label(label: str) -> str:
    token = label.strip().splitlines()[0].split()[0].lower()
    if token in _LOCALE:
        return _LOCALE[token]
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
        raise ValueError("English to English is not supported. One language must be other than English.")
    return source_name, target_name
