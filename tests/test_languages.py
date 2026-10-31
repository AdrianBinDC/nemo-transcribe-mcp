import pytest

from nemotron_transcribe_mcp.languages import (
    apply_detected_source,
    language_id_note,
    supported_languages,
    translation_pair,
)


def test_french_from_english_accepts_a_name_or_a_code() -> None:
    assert translation_pair("French") == ("English", "French")
    assert translation_pair("fr", "en") == ("English", "French")


def test_two_non_english_languages_are_rejected() -> None:
    with pytest.raises(ValueError, match="English"):
        translation_pair("French", "German")


def test_a_language_outside_the_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="destination Turkish") as raised:
        translation_pair("Turkish", "Polish")
    assert "source Polish is set" in str(raised.value)


def test_a_heard_language_overrides_the_default_source() -> None:
    settings: dict[str, object] = {"target_language": "English", "fallback_source": "de"}
    apply_detected_source(settings, "fr_fr")
    assert settings["source_language"] == "French"


def test_a_missed_detection_keeps_the_default_source() -> None:
    settings: dict[str, object] = {"target_language": "English", "fallback_source": "pl"}
    apply_detected_source(settings, "")
    assert settings["source_language"] == "Polish"


def test_supported_languages_names_the_card_level() -> None:
    rows = {row["code"]: row for row in supported_languages()}
    assert len(rows) == 41
    assert rows["fr"]["level"] == "leading"
    assert rows["fr"]["translate"] == "yes"
    assert rows["ja"]["level"] == "production_ready"
    assert rows["ar"]["level"] == "supported"
    assert rows["tr"]["translate"] == "no"
    assert rows["lv"]["identify"] == "low"
    assert rows["bn"]["identify"] == "no"
    assert sum(row["level"] == "leading" for row in rows.values()) == 13


def test_latvian_and_slovenian_labels_carry_a_warning() -> None:
    assert language_id_note("en_us") == "en_us"
    assert "Low confidence" in language_id_note("lv_lv")
    assert "Low confidence" in language_id_note("sl")
