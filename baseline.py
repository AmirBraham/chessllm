"""Baseline: how good are the model's chess moves, measured in centipawns?

The headline is mean cp_loss -- how much the chosen move gives up against
Stockfish's best. It is reported next to two references measured on the same
positions, because the number is meaningless alone:

    random legal move   ~377 cp   the floor to beat
    Stockfish's own     ~10 cp    the ceiling (nonzero: the search is one ply
                                  deeper after the move than at the root)

Top-1 agreement with Stockfish is deliberately not reported. On these
positions a random move scores 1% against a 3.4% chance rate -- the metric has
no resolution at this skill level.

Run this on a GPU box; batched generation over a few hundred positions is slow
on a laptop.
"""

import argparse
import json
import random
import statistics as st
from pathlib import Path

import chess

from board import build_prompt, parse_move
from engine import Engine
from qwen3 import MODEL_ID, generate, load

GOOD_MOVE_CP = 50  # "reasonable move" threshold


def score(engine, board, move):
    """One result row. `move` is a legal chess.Move, or None."""
    return {
        "legal": move is not None,
        "cp_loss": engine.cp_loss(board, move) if move else None,
    }


def run_random(engine, boards, seed=0):
    rng = random.Random(seed)
    return [score(engine, b, rng.choice(list(b.legal_moves))) for b in boards]


def run_stockfish(engine, boards):
    return [score(engine, b, engine.best(b)[0]) for b in boards]


def run_model(model, tok, engine, boards, histories, include_legal_moves,
              max_new_tokens, batch_size, think):
    prompts = [
        build_prompt(b, h, include_legal_moves)
        for b, h in zip(boards, histories)
    ]
    texts, lengths = generate(
        model, tok, prompts,
        max_new_tokens=max_new_tokens, think=think, batch_size=batch_size,
    )

    rows = []
    for board, text, length in zip(boards, texts, lengths):
        move, raw = parse_move(board, text)
        row = score(engine, board, move)
        # "answered" means it committed to something inside \boxed{}. A
        # completion that ran out of budget mid-reasoning has no answer, so it
        # contributes to no cp_loss average.
        row["answered"] = raw is not None
        row["tokens"] = length
        row["truncated"] = length >= max_new_tokens
        # Kept so the run can be read afterwards, not just scored. A number
        # without the completion behind it cannot tell you *why* the model
        # failed -- whether it misread the position, named an illegal move, or
        # never got to an answer.
        row["fen"] = board.fen()
        row["move"] = board.san(move) if move else None
        row["raw"] = raw
        row["completion"] = text
        rows.append(row)

    return rows


def summarize(name, rows):
    n = len(rows)
    losses = [r["cp_loss"] for r in rows if r["cp_loss"] is not None]

    print(f"\n{name}  (n={n})")
    if not losses:
        print("  no legal moves produced")
    else:
        good = sum(1 for x in losses if x <= GOOD_MOVE_CP) / len(losses)
        print(f"  mean cp_loss     {st.mean(losses):6.0f}")
        print(f"  median cp_loss   {st.median(losses):6.0f}")
        print(f"  good moves <={GOOD_MOVE_CP}cp  {good:6.1%}")

    if "answered" in rows[0]:
        rate = lambda k: sum(1 for r in rows if r[k]) / n  # noqa: E731
        print(f"  answered         {rate('answered'):6.1%}")
        print(f"  legal rate       {rate('legal'):6.1%}")
        print(f"  truncated        {rate('truncated'):6.1%}")
        print(f"  mean tokens      {st.mean(r['tokens'] for r in rows):6.0f}")
    return {"name": name, "n": n, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", default="data/positions.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="write raw rows to JSON")
    # 48 not 16: 4B leaves ~15GB of a 24GB card free, and generation is
    # bandwidth bound -- a bigger batch amortises the weight read across
    # more sequences, which is the actual bottleneck.
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                        help="vllm uses continuous batching (much faster, "
                             "but installs its own pinned torch)")
    # No local probe was possible, so this default is a guess. The truncation
    # rate in the output tells you whether it was big enough: if it is not
    # near zero, raise this or pass --no-think.
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-think", action="store_true",
                        help="disable Qwen3 thinking mode (much shorter, cheaper)")
    parser.add_argument("--no-legal-moves", action="store_true",
                        help="drop the legal-move list from the prompt")
    parser.add_argument("--model", default=MODEL_ID,
                        help="HF model id, e.g. Qwen/Qwen3-4B")
    parser.add_argument("--skip-refs", action="store_true",
                        help="skip the random and Stockfish reference rows")
    args = parser.parse_args()

    with open(args.positions) as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    if "moves" not in records[0]:
        raise SystemExit(
            f"{args.positions} has no 'moves' field -- the prompt is built "
            "from movetext now. Regenerate: uv run python build_data.py"
        )

    boards = [chess.Board(r["fen"]) for r in records]
    histories = [r["moves"] for r in records]
    print(f"{len(boards)} positions from {args.positions}")

    results = []
    with Engine(depth=12, threads=1) as engine:
        if not args.skip_refs:
            print("\nreferences...")
            results.append(summarize("random legal move", run_random(engine, boards)))
            results.append(summarize("stockfish best", run_stockfish(engine, boards)))

        label = (
            f"{args.model.split('/')[-1]} "
            f"legal={not args.no_legal_moves} think={not args.no_think}"
        )
        print("\nloading model...")
        model, tok = load(args.model, backend=args.backend)
        print(f"generating ({label})...")
        rows = run_model(
            model, tok, engine, boards, histories,
            include_legal_moves=not args.no_legal_moves,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            think=not args.no_think,
        )
        results.append(summarize(label, rows))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(results, f, indent=1)
        # No plotting here on purpose: the pod produces data, charts are made
        # locally in probe.ipynb. Keeps matplotlib off the pod and keeps GPU
        # time spent on generation.
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
