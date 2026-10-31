from nemotron_transcribe_mcp.plan import Silence, plan_pieces, stitch_transcripts


def test_cuts_at_the_longest_pause_in_the_last_fifth() -> None:
    # 55 min cap. The window is 44:00–55:00 (2640–3300 s).
    silences = [
        Silence(2490, 2510),  # before the window
        Silence(2690, 2710),  # 20 s, inside the window
        Silence(3060, 3140),  # 80 s, the longest inside the window
    ]

    pieces = plan_pieces(6000, silences, hard_max_s=3300, speakers=False)

    assert [(piece.start_s, piece.end_s) for piece in pieces] == [
        (0, 3100),
        (3100, 6000),
    ]


def test_hard_cut_repeats_five_seconds_when_no_pause_is_in_the_window() -> None:
    pieces = plan_pieces(6000, [], hard_max_s=3300, speakers=False)

    # 0.9 × 3300 = 2970. The next piece starts 5 s earlier.
    assert [(piece.start_s, piece.end_s) for piece in pieces] == [
        (0, 2970),
        (2965, 6000),
    ]


def test_speaker_job_repeats_ten_seconds_at_the_join() -> None:
    silences = [Silence(3060, 3140)]

    pieces = plan_pieces(6000, silences, hard_max_s=3300, speakers=True)

    assert [(piece.start_s, piece.end_s) for piece in pieces] == [
        (0, 3100),
        (3090, 6000),
    ]


def test_stitch_drops_words_repeated_at_the_join() -> None:
    assert (
        stitch_transcripts(["alpha bravo charlie", "bravo charlie delta"])
        == "alpha bravo charlie delta"
    )


def test_stitch_keeps_a_newline_when_the_join_does_not_repeat() -> None:
    assert stitch_transcripts(["first piece.", "second piece."]) == "first piece.\nsecond piece."
