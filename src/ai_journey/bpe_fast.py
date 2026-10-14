"""Heap-based BPE encoding with the reference encoder's exact rank semantics."""

from heapq import heappop, heappush

from .bpe import ByteTokenizer, RegexTokenizer


def _encode_piece(tokenizer, text):
    values = list(text.encode("utf-8"))
    size = len(values)
    if size < 2:
        return tuple(values)
    previous = [index - 1 for index in range(size)]
    following = [index + 1 for index in range(size)]
    following[-1] = -1
    alive = [True] * size
    ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}
    heap = []

    def offer(left):
        if left < 0 or not alive[left]:
            return
        right = following[left]
        if right < 0:
            return
        pair = (values[left], values[right])
        if pair in ranks:
            heappush(heap, (ranks[pair], left, right, *pair))

    for index in range(size - 1):
        offer(index)
    while heap:
        rank, left, right, first, second = heappop(heap)
        if (
            not alive[left]
            or not alive[right]
            or following[left] != right
            or (values[left], values[right]) != (first, second)
        ):
            continue
        values[left] = 256 + rank
        alive[right] = False
        following[left] = following[right]
        if following[right] >= 0:
            previous[following[right]] = left
        offer(previous[left])
        offer(left)
    return tuple(value for index, value in enumerate(values) if alive[index])


def encode_fast(tokenizer: ByteTokenizer, text: str) -> tuple[int, ...]:
    if not isinstance(tokenizer, ByteTokenizer) or not isinstance(text, str):
        raise TypeError("fast encoding requires a byte tokenizer and text")
    pieces = (
        tokenizer.pieces(text) if isinstance(tokenizer, RegexTokenizer) else (text,)
    )
    return tuple(token for piece in pieces for token in _encode_piece(tokenizer, piece))
