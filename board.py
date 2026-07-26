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


def build_prompt(board, include_board=True, include_legal_moves=True):
    """Build the model prompt. The two flags define the 2x2 baseline grid."""
    side = "White" if board.turn == chess.WHITE else "Black"

    parts = [
        "You are a chess engine. Choose the best move.\n",
        f"FEN: {board.fen()}\n",
    ]
    if include_board:
        parts.append(f"{ascii_board(board)}\n")

    context = f"{side} to move.\n"
    if include_legal_moves:
        context += f"Legal moves: {' '.join(legal_moves_san(board))}\n"
    parts.append(context)

    parts.append(
        "Think briefly, then write the move on the last line "
        "in standard algebraic notation (SAN)."
    )
    return "\n".join(parts)


def parse_move(board, completion):
    """Extract the move the completion settles on, scanning after </think>.

    Returns (Move, raw) if legal, (None, raw) if illegal, (None, None) if no
    move-shaped text at all -- the reward function scores those differently.
    """
    tail = completion.rsplit("</think>", 1)[-1]
    if not tail.strip():
        tail = completion

    fallback = None
    for line in reversed([ln for ln in tail.splitlines() if ln.strip()]):
        for raw in reversed(SAN_TOKEN.findall(line)):
            fallback = fallback or raw
            try:
                return board.parse_san(raw), raw
            except ValueError:
                pass
        for raw in reversed(UCI_TOKEN.findall(line)):
            fallback = fallback or raw
            try:
                move = chess.Move.from_uci(raw)
            except ValueError:
                continue
            if move in board.legal_moves:
                return move, raw

    return None, fallback
