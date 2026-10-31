from dataclasses import dataclass


@dataclass(frozen=True)
class Silence:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def mid_s(self) -> float:
        return (self.start_s + self.end_s) / 2


@dataclass(frozen=True)
class Piece:
    start_s: float
    end_s: float
    follows_hard_cut: bool = False


def plan_pieces(
    duration_s: float,
    silences: list[Silence],
    *,
    hard_max_s: float,
    speakers: bool,
) -> list[Piece]:
    cuts: list[tuple[float, str]] = []
    start = 0.0
    while duration_s - start > hard_max_s:
        window_lo = start + 0.8 * hard_max_s
        window_hi = start + hard_max_s
        window = [item for item in silences if window_lo <= item.mid_s <= window_hi]
        if window:
            cut = max(window, key=lambda item: item.duration_s).mid_s
            kind = "silence"
        else:
            cut = start + 0.9 * hard_max_s
            kind = "hard"
        cuts.append((cut, kind))
        start = cut
    edges = [0.0, *[cut for cut, _ in cuts], duration_s]
    pieces: list[Piece] = []
    for index in range(len(edges) - 1):
        piece_start = edges[index]
        hard = index > 0 and cuts[index - 1][1] == "hard"
        if hard:
            piece_start -= 5
        elif index > 0 and speakers:
            piece_start -= 10
        pieces.append(Piece(max(0.0, piece_start), edges[index + 1], hard))
    return pieces


def stitch_transcripts(parts: list[str]) -> str:
    joined = ""
    for part in parts:
        text = part.strip()
        if not text:
            continue
        if not joined:
            joined = text
            continue
        left = joined.split()
        right = text.split()
        limit = min(30, len(left), len(right))
        overlap = 0
        for size in range(limit, 0, -1):
            if left[-size:] == right[:size]:
                overlap = size
                break
        if overlap:
            rest = " ".join(right[overlap:])
            joined = f"{joined} {rest}".rstrip()
        else:
            joined = f"{joined}\n{text}"
    return joined
