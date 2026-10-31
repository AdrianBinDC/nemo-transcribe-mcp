import math

CONTEXT_TOKENS = 65536
TOKEN_PAD = 512
AUDIO_TOKENS_PER_SECOND = 12.5
# The model thinks with 16,384 tokens. Audio costs 12.5 tokens per second,
# so a question can cover 16384 / 12.5 = 1,311 seconds, about 22 minutes.
REASONING_BUDGET_TOKENS = 16384


def audio_token_count(duration_s: float) -> int:
    return math.ceil(duration_s * AUDIO_TOKENS_PER_SECOND)


def question_span(
    duration_s: float,
    start_s: float | None,
    end_s: float | None,
) -> tuple[float, float]:
    if (start_s is None) != (end_s is None):
        raise ValueError("Pass both start and end, or neither.")
    if start_s is None:
        start, end = 0.0, duration_s
    else:
        start, end = float(start_s), float(end_s)
        if start < 0 or end <= start:
            raise ValueError("end must be after start")
        if end > duration_s:
            raise ValueError("end is past the end of the file")
    if audio_token_count(end - start) > REASONING_BUDGET_TOKENS:
        raise ValueError(
            "This recording is too long to answer in one pass. Pass start and end for the part you want."
        )
    return start, end


def output_token_cap(duration_s: float) -> int:
    audio_tokens = audio_token_count(duration_s)
    room = CONTEXT_TOKENS - audio_tokens - TOKEN_PAD
    if room <= 0:
        raise ValueError(
            f"audio is too long for one request ({duration_s:.1f} s, {audio_tokens} audio tokens)"
        )
    return room
