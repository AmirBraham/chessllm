import re

import chess

SAN_TOKEN = re.compile(
    r"O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
)
UCI_TOKEN = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")


def ascii_board(board):
    """8x8 grid, White at the bottom. Uppercase = White, lowercase = Black."""
    header = "  a b c d e f g h"
    lines = [header]
    for rank in range(7, -1, -1):
        cells = [
            (p.symbol() if (p := board.piece_at(chess.square(f, rank))) else ".")
            for f in range(8)
        ]
        lines.append(f"{rank + 1} " + " ".join(cells) + f" {rank + 1}")
    lines.append(header)
    return "\n".join(lines)


def legal_moves_san(board):
    """Legal moves in SAN, sorted so the prompt is deterministic."""
    return sorted(board.san(m) for m in board.legal_moves)


def format_history(moves_san):
    """Moves as PGN movetext: '1. e4 e5 2. Nf3 Nc6 ...'"""
    out = []
    for i, san in enumerate(moves_san):
        if i % 2 == 0:
            out.append(f"{i // 2 + 1}.")
        out.append(san)
    return " ".join(out)


def build_prompt(board, moves_san, include_legal_moves=True):
    r"""Build the model prompt from the game's move history.

    The position is given as movetext, not as a FEN or a diagram. Both of
    those were tried and failed: a 0.6B model decodes `2b2rk1` as "two kings,
    two rooks, and a bishop" and burns its whole token budget doing it, while
    ignoring an ASCII grid printed directly below. Movetext is the one board
    encoding it saw at scale during pretraining, so whatever chess knowledge
    it has should be reachable this way.
    """
    side = "White" if board.turn == chess.WHITE else "Black"

    parts = [
        "You are a chess engine. Choose the best move.\n",
        f"Game so far:\n{format_history(moves_san)}\n",
        f"{side} to move.",
    ]
    if include_legal_moves:
        parts.append(f"Legal moves: {' '.join(legal_moves_san(board))}")

    parts.append(
        "Give your move in standard algebraic notation (SAN) as: "
        r"\boxed{MOVE}"
    )
    return "\n".join(parts)


def extract_boxed(text):
    r"""Contents of the last \boxed{...}, or None.

    Ported from the book's `get_last_boxed` (ch03). Counts brace depth rather
    than matching the first "}", so nested braces survive. Unbalanced braces
    return None -- that means the completion was cut off mid-answer, which is
    not an answer.
    """
    start = text.rfind(r"\boxed")
    if start == -1:
        return None

    i = start + len(r"\boxed")
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None

    i += 1
    depth, content_start = 1, i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None
    return text[content_start:i - 1].strip()


def _to_move(board, raw):
    """SAN or UCI string -> legal Move, or None."""
    try:
        return board.parse_san(raw)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(raw)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def parse_move(board, completion, strict=True):
    r"""Extract the move the completion commits to.

    Returns (Move, raw) if legal, (None, raw) if a move was named but illegal,
    (None, None) if nothing was named -- three outcomes the reward scores
    differently.

    strict=True reads only \boxed{...}. This matters: without it, a completion
    that rambles past its token budget still yields a move, because the scan
    picks up any move-shaped token in the prose. Rewarding that during GRPO
    would teach the model to produce text that games the parser rather than
    good chess.

    strict=False keeps the old prose scan, for measuring how much of a
    reported legal rate was parser artifact.
    """
    boxed = extract_boxed(completion)
    if boxed is not None:
        move = _to_move(board, boxed)
        if move is not None:
            return move, boxed
        # Answered, but wrapped: "**Ng5**", "Ng5 is best", ...
        for raw in SAN_TOKEN.findall(boxed) + UCI_TOKEN.findall(boxed):
            move = _to_move(board, raw)
            if move is not None:
                return move, raw
        return None, boxed

    if strict:
        return None, None

    tail = completion.rsplit("</think>", 1)[-1]
    if not tail.strip():
        tail = completion

    fallback = None
    for line in reversed([ln for ln in tail.splitlines() if ln.strip()]):
        for raw in SAN_TOKEN.findall(line) + UCI_TOKEN.findall(line):
            fallback = fallback or raw
            move = _to_move(board, raw)
            if move is not None:
                return move, raw

    return None, fallback
