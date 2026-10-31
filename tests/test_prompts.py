from nemotron_transcribe_mcp.prompts import (
    language_id_prompt,
    prompt_for,
    system_for,
    translation_prompt,
    verbatim_prompt,
)

CARD_VERBATIM = (
    "Audio transcription. Language: as spoken. "
    "Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. "
    "Output: transcript only."
)

CARD_CLEAN = (
    "Audio transcription. Language: as spoken. "
    "Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. "
    "Formatting: punctuation and capitalization. "
    "Output: transcript only."
)

CARD_TECHNICAL = (
    "Transcribe as a clean written transcript. "
    "Preserve exact values and punctuation/separators in emails, URLs, file paths, commands, and identifiers. "
    "Use conventional written forms."
)

CARD_RAW = (
    "Transcribe word for word without punctuation. "
    "Write numbers as spoken words and keep individually spoken letters separate."
)


def test_verbatim_prompt_matches_the_model_card() -> None:
    assert verbatim_prompt() == CARD_VERBATIM


def test_clean_prompt_matches_the_model_card() -> None:
    assert prompt_for("clean") == CARD_CLEAN


def test_technical_prompt_matches_the_model_card() -> None:
    assert prompt_for("technical") == CARD_TECHNICAL


def test_raw_dictation_prompt_matches_the_model_card() -> None:
    assert prompt_for("raw_dictation") == CARD_RAW


def test_translation_prompt_names_the_target_language() -> None:
    assert translation_prompt("French") == "Please translate the given speech to French."


def test_fleurs_language_id_prompt_matches_the_model_card() -> None:
    prompt = language_id_prompt("fleurs")
    assert prompt.startswith("Identify the language spoken in this audio.")
    assert "en_us" in prompt
    assert prompt.count(",") == 35


def test_audio_question_system_prompt_matches_the_model_card() -> None:
    assert system_for("audio_qa") == (
        "Answer the question directly using evidence from every provided audio clip in order. "
        "Answer every requested part, preserve exact names and numbers, "
        "and give a concise, specific answer first. Do not invent details."
    )


def test_caller_labels_use_the_short_card_prompt() -> None:
    assert language_id_prompt("fleurs", ["en", "fr"]) == (
        "Identify the spoken language. Respond with exactly one label from en, fr and nothing else."
    )
