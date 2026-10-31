import math

CONTEXT_TOKENS = 65536
TOKEN_PAD = 512
AUDIO_TOKENS_PER_SECOND = 12.5


def output_token_cap(duration_s: float) -> int:
    audio_tokens = math.ceil(duration_s * AUDIO_TOKENS_PER_SECOND)
    room = CONTEXT_TOKENS - audio_tokens - TOKEN_PAD
    if room <= 0:
        raise ValueError(
            f"audio is too long for one request ({duration_s:.1f} s, {audio_tokens} audio tokens)"
        )
    return room
