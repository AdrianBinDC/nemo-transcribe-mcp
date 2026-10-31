import pytest

from nemo_transcribe_mcp.tokens import output_token_cap, question_span


def test_ten_second_clip_reserves_transcript_room() -> None:
    # 10 s × 12.5 = 125 audio tokens. 65536 - 125 - 512 = 64899.
    assert output_token_cap(10) == 64899


def test_one_second_uses_the_audio_ceiling() -> None:
    # ceil(12.5) = 13. 65536 - 13 - 512 = 65011.
    assert output_token_cap(1) == 65011


def test_a_question_uses_the_whole_file_when_it_fits() -> None:
    assert question_span(60, None, None) == (0.0, 60.0)


def test_a_question_keeps_the_requested_part() -> None:
    assert question_span(2000, 10, 70) == (10.0, 70.0)


def test_a_question_about_a_long_file_asks_for_a_part() -> None:
    with pytest.raises(ValueError, match="start and end"):
        question_span(2000, None, None)


def test_file_longer_than_the_context_window_does_not_fit() -> None:
    with pytest.raises(ValueError, match="too long"):
        output_token_cap(90 * 60)
