import pytest

from nemotron_transcribe_mcp.languages import translation_pair


def test_french_from_english_accepts_a_name_or_a_code() -> None:
    assert translation_pair("French") == ("English", "French")
    assert translation_pair("fr", "en") == ("English", "French")


def test_two_non_english_languages_are_rejected() -> None:
    with pytest.raises(ValueError, match="English"):
        translation_pair("French", "German")


def test_a_language_outside_the_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="Turkish"):
        translation_pair("Turkish")
