from nemotron_transcribe_mcp.prompts import prompt_for, verbatim_prompt

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
