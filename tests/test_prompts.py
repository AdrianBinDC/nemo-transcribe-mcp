from nemotron_transcribe_mcp.prompts import verbatim_prompt

CARD_VERBATIM = (
    "Audio transcription. Language: as spoken. "
    "Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. "
    "Output: transcript only."
)


def test_verbatim_prompt_matches_the_model_card() -> None:
    assert verbatim_prompt() == CARD_VERBATIM
