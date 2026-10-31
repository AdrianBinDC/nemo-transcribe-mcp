import pytest

from nemotron_transcribe_mcp.tokens import output_token_cap


def test_ten_second_clip_reserves_transcript_room() -> None:
    # 10 s × 12.5 = 125 audio tokens. 65536 - 125 - 512 = 64899.
    assert output_token_cap(10) == 64899


def test_one_second_uses_the_audio_ceiling() -> None:
    # ceil(12.5) = 13. 65536 - 13 - 512 = 65011.
    assert output_token_cap(1) == 65011


def test_file_longer_than_the_context_window_does_not_fit() -> None:
    with pytest.raises(ValueError, match="too long"):
        output_token_cap(90 * 60)
