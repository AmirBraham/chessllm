"""Build a position dataset from real human games.

Positions come from `angeluriot/chess_games` on Hugging Face (14.2M games,
mean Elo 2388). Real games matter: the model learned chess from PGN text
during pretraining, so human positions are in-distribution in a way engine
self-play is not.

The file is a single 7.3 GB parquet, but it has 14,189 row groups of 1,000
games and we need 3 of its 9 columns -- so reading scattered row groups over
HTTP range requests costs a few MB, not 7.3 GB.

Positions are filtered for *reward spread*. Where every move is within 20 cp,
all GRPO rollouts score alike, advantages collapse to zero, and the step
teaches nothing.

The dataset stores no engine evaluations -- only the position and some notes
about how it was chosen. Stockfish is the single source of truth at scoring
time, so a cached number can never disagree with it.
"""

import argparse
import json
import random
from pathlib import Path

import chess
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from engine import Engine

PARQUET = (
    "datasets/angeluriot/chess_games"
    "@refs/convert/parquet/default/train/0000.parquet"
)

DEPTH = 12  # same depth the reward uses
MIN_ELO = 2000
MIN_PLIES = 20
MIN_LEGAL = 6  # fewer and the choice is trivial
MIN_SPREAD = 50  # cp between best and 5th-best
MAX_ABS_EVAL = 800  # already decided; move quality is moot
OPENING_PLIES = 8  # skip memorised theory
ENDING_PLIES = 4  # skip forced final sequences


def iter_games(seed):
    """Yield (moves_san, [white_elo, black_elo]) from random row groups."""
    parquet = pq.ParquetFile(HfFileSystem().open(PARQUET, "rb"))
    groups = list(range(parquet.metadata.num_row_groups))
    random.Random(seed).shuffle(groups)

    for group in groups:
        table = parquet.read_row_group(
            group, columns=["moves_san", "white_elo", "black_elo"]
        )
        for row in table.to_pylist():
            white, black = row["white_elo"], row["black_elo"]
            moves = row["moves_san"]
            if white is None or black is None or moves is None:
                continue
            if min(white, black) >= MIN_ELO and len(moves) >= MIN_PLIES:
                yield moves, [white, black]


def sample_position(moves_san, rng):
    """Replay a game to a random ply. Returns (board, ply) or None."""
    low, high = OPENING_PLIES, len(moves_san) - ENDING_PLIES
    if high <= low:
        return None

    ply = rng.randrange(low, high)
    board = chess.Board()
    try:
        for san in moves_san[:ply]:
            board.push_san(san)
    except ValueError:
        return None  # malformed SAN in the source data
    return board, ply


def build(n, out, seed=0):
    rng = random.Random(seed)
    records, seen = [], set()
    considered = 0

    with Engine(depth=DEPTH, threads=1) as engine:
        for moves_san, elo in iter_games(seed):
            if len(records) >= n:
                break
            considered += 1

            sampled = sample_position(moves_san, rng)
            if sampled is None:
                continue
            board, ply = sampled

            if board.is_game_over() or board.fen() in seen:
                continue

            n_legal = board.legal_moves.count()
            if n_legal < MIN_LEGAL:
                continue

            tops = engine.top_moves(board, k=5)
            best_cp = tops[0][1]
            spread = best_cp - tops[-1][1]
            if abs(best_cp) > MAX_ABS_EVAL or spread < MIN_SPREAD:
                continue

            seen.add(board.fen())
            records.append(
                {
                    "fen": board.fen(),
                    # The prompt is built from movetext, not the FEN. The FEN
                    # is kept so a position can be reconstructed in one step
                    # without replaying the game.
                    "moves": moves_san[:ply],
                    "spread_cp": spread,
                    "n_legal": n_legal,
                    "ply": ply,
                    "elo": elo,
                }
            )
            if len(records) % 50 == 0:
                print(f"  {len(records)}/{n} kept of {considered} games")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f, indent=1)
    print(f"\nwrote {len(records)} positions to {out} ({considered} games seen)")
    return records


def main():
    parser = argparse.ArgumentParser(description="Build a chess position dataset.")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", default="data/positions.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.n, args.out, args.seed)


if __name__ == "__main__":
    main()
