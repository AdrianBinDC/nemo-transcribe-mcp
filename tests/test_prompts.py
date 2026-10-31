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


def test_verbatim_prompt_matches_the_model_card() -> None:
    assert verbatim_prompt() == CARD_VERBATIM


def test_clean_prompt_matches_the_model_card() -> None:
    assert prompt_for("clean") == CARD_CLEAN
