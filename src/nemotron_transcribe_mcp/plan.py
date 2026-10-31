import re
from dataclasses import dataclass
from itertools import permutations

from nemotron_transcribe_mcp.logsetup import get_logger

_SPEAKER = re.compile(r"<spk:(\d)>")
_LOW_CONFIDENCE = 0.6
log = get_logger("plan")


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


def speech_window(
    duration_s: float,
    silences: list[Silence],
    *,
    length_s: float = 10,
) -> tuple[float, float]:
    if duration_s <= 0:
        raise ValueError("clip has no audio")
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for silence in silences:
        if silence.start_s > cursor:
            spans.append((cursor, min(silence.start_s, duration_s)))
        cursor = max(cursor, silence.end_s)
    if cursor < duration_s:
        spans.append((cursor, duration_s))
    usable = [span for span in spans if span[1] - span[0] >= 1]
    start, end = (usable or spans or [(0.0, duration_s)])[0]
    return start, min(end, start + length_s)


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
        if _SPEAKER.search(joined) and _SPEAKER.search(text):
            joined = _stitch_speakers(joined, text)
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


def _stream(text: str) -> list[tuple[str | None, str]]:
    speaker: str | None = None
    rows: list[tuple[str | None, str]] = []
    for token in text.split():
        match = _SPEAKER.fullmatch(token)
        if match:
            speaker = match.group(1)
            continue
        rows.append((speaker, token))
    return rows


def _overlap_size(left: list[tuple[str | None, str]], right: list[tuple[str | None, str]]) -> int:
    limit = min(30, len(left), len(right))
    for size in range(limit, 0, -1):
        if [word for _, word in left[-size:]] == [word for _, word in right[:size]]:
            return size
    return 0


def _assignment(
    left: list[tuple[str | None, str]],
    right: list[tuple[str | None, str]],
    size: int,
) -> tuple[dict[str, str] | None, float]:
    if size == 0:
        return None, 0.0
    counts: dict[tuple[str, str], int] = {}
    for (left_speaker, _), (right_speaker, _) in zip(left[-size:], right[:size], strict=True):
        if left_speaker is None or right_speaker is None:
            continue
        key = (right_speaker, left_speaker)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return None, 0.0
    rights = sorted({pair[0] for pair in counts})
    lefts = sorted({pair[1] for pair in counts})
    best_score = -1
    best_maps: list[dict[str, str]] = []
    if len(lefts) >= len(rights):
        choices = permutations(lefts, len(rights))
        for choice in choices:
            mapping = dict(zip(rights, choice, strict=True))
            score = sum(counts.get((speaker, mapping[speaker]), 0) for speaker in rights)
            if score > best_score:
                best_score = score
                best_maps = [mapping]
            elif score == best_score:
                best_maps.append(mapping)
    else:
        for chosen in permutations(rights, len(lefts)):
            mapping = dict(zip(chosen, lefts, strict=True))
            score = sum(counts.get((speaker, mapping[speaker]), 0) for speaker in chosen)
            if score > best_score:
                best_score = score
                best_maps = [mapping]
            elif score == best_score:
                best_maps.append(mapping)
    if len(best_maps) != 1:
        return None, 0.0
    mapping = {
        speaker: partner
        for speaker, partner in best_maps[0].items()
        if counts.get((speaker, partner), 0)
    }
    if set(mapping) != set(rights):
        settled = sum(counts[(speaker, mapping[speaker])] for speaker in mapping)
        return None, settled / total
    for speaker, partner in mapping.items():
        chosen = counts[(speaker, partner)]
        if any(counts.get((speaker, other), 0) >= chosen for other in lefts if other != partner):
            return None, 0.0
    confidence = sum(counts[(speaker, mapping[speaker])] for speaker in mapping) / total
    if confidence < _LOW_CONFIDENCE:
        return None, confidence
    return mapping, confidence


def _fresh(used: set[str], mapping: dict[str, str]) -> str:
    taken = used | set(mapping.values())
    for number in range(8):
        label = str(number)
        if label not in taken:
            return label
    raise ValueError("more than 8 speakers")


def _render(
    stream: list[tuple[str | None, str]],
    mapping: dict[str, str] | None,
    used: set[str],
) -> str:
    tokens: list[str] = []
    current: str | None = None
    for speaker, word in stream:
        if speaker is None:
            tokens.append(word)
            current = None
            continue
        if mapping is None:
            label = speaker
        else:
            label = mapping.get(speaker)
            if label is None:
                label = _fresh(used, mapping)
                mapping[speaker] = label
        if label != current:
            tokens.append(f"<spk:{label}>")
            current = label
        tokens.append(word)
    return " ".join(tokens)


def _stitch_speakers(left_text: str, right_text: str) -> str:
    left = _stream(left_text)
    right = _stream(right_text)
    size = _overlap_size(left, right)
    mapping, confidence = _assignment(left, right, size)
    remainder = right[size:]
    if mapping is None:
        log.warning("speaker_join_uncertain", confidence=round(confidence, 2))
        rest = _render(remainder, None, set())
        marker = f"[speaker-join-uncertain {confidence:.2f}]"
        if rest:
            return f"{left_text}\n{marker}\n{rest}"
        return f"{left_text}\n{marker}"
    log.info("speaker_map", confidence=round(confidence, 2))
    rest = _render(remainder, dict(mapping), set(_SPEAKER.findall(left_text)))
    if not rest:
        return left_text
    return f"{left_text} {rest}".rstrip()
