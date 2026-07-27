"""Does the model know enough chess for RL to have anything to amplify?

    python probe.py --n 20 --out runs/probe.json

RL boosts responses the model already produces sometimes; it does not install
capabilities it lacks. DeepSeek's own phrasing: "the improvement is attributed
to boosting the correct response from top K rather than the enhancement of
fundamental capabilities." A published GRPO run on 8B models spent days
converging on "always push the a-pawn" because the model could not read a
board -- and the giveaway was that it could not say which squares a knight on
b6 reaches.

Two probes, both with mechanically checkable answers:

  moves   -- where does a piece move from an empty square? Tests whether the
             model knows the rules at all.
  board   -- after this movetext, what is on <square>? Tests whether it can
             track state, which is what our actual prompt demands.

If `moves` is near zero, no amount of GRPO will produce chess, and the budget
is better spent on distillation.
"""

import argparse
import json
import random
import re
from pathlib import Path

import chess

from qwen3 import MODEL_ID, generate, load

SQUARE = re.compile(r"\b[a-h][1-8]\b")
PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}
PROBE_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def boxed(text):
    """Contents of the last \\boxed{...}, else the last non-empty line.

    Uses board.extract_boxed, which counts brace depth. Stopping at the first
    "}" turns the common \\boxed{\\text{empty}} into "\\text{empty" -- which
    happens to grade right here, but is exactly the kind of sloppy parsing
    that has already produced two rounds of fake results.
    """
    from board import extract_boxed

    inner = extract_boxed(text)
    if inner is not None:
        # Strip one LaTeX wrapper, e.g. \text{empty} -> empty
        match = re.fullmatch(r"\\[a-z]+\{(.*)\}", inner.strip(), re.S)
        return match.group(1) if match else inner

    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def moves_probe(n, rng):
    """Where can a lone piece move on an empty board?

    Pieces are cycled rather than sampled, so each gets n/5 questions. Drawing
    at random left 6 knights and 2 bishops in a 25-item run, which makes any
    per-piece breakdown meaningless.
    """
    items = []
    for index in range(n):
        piece = PROBE_PIECES[index % len(PROBE_PIECES)]
        square = rng.randrange(64)
        name = PIECE_NAMES[piece]

        board = chess.Board(None)
        board.set_piece_at(square, chess.Piece(piece, chess.WHITE))
        answer = {chess.square_name(s) for s in board.attacks(square)}

        items.append({
            "kind": "moves",
            "prompt": (
                f"An otherwise empty chess board has a single white {name} "
                f"on {chess.square_name(square)}.\n"
                f"List every square that {name} can move to.\n"
                r"Put the squares, separated by spaces, inside \boxed{...}"
            ),
            "answer": sorted(answer),
        })
    return items


def board_probe(records, n, rng):
    """After this movetext, what sits on a given square?

    Exactly half the squares are empty, alternating. A coin flip per item made
    the always-answer-"empty" baseline drift with the sample: it came out
    40% in one 25-item run, so a model scoring 40% looked like it was tracking
    when it was answering "empty" to everything.
    """
    items = []
    pool = records * (n // len(records) + 1)
    for index, record in enumerate(rng.sample(pool, n) if n > len(records)
                                   else rng.sample(records, n)):
        board = chess.Board()
        for san in record["moves"]:
            board.push_san(san)

        occupied = [s for s in chess.SQUARES if board.piece_at(s)]
        empty = [s for s in chess.SQUARES if not board.piece_at(s)]
        square = rng.choice(empty if index % 2 else occupied)

        piece = board.piece_at(square)
        answer = (
            "empty" if piece is None
            else f"{'white' if piece.color else 'black'} {PIECE_NAMES[piece.piece_type]}"
        )

        from board import format_history
        items.append({
            "kind": "board",
            "prompt": (
                f"Game so far:\n{format_history(record['moves'])}\n\n"
                f"What is on square {chess.square_name(square)}?\n"
                'Answer "empty", or the colour and piece, e.g. "black knight".\n'
                r"Put the answer inside \boxed{...}"
            ),
            "answer": answer,
        })
    return items


def grade(item, completion):
    """1.0 correct, 0.0 wrong. Partial credit for `moves` would hide the
    difference between knowing the rule and guessing near it."""
    said = boxed(completion).strip().lower()

    if item["kind"] == "moves":
        return float(set(SQUARE.findall(said)) == set(item["answer"]))

    expected = item["answer"]
    if expected == "empty":
        return float("empty" in said and not any(
            name in said for name in PIECE_NAMES.values()
        ))
    colour, name = expected.split()
    return float(colour in said and name in said)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="items per probe")
    parser.add_argument("--positions", default="data/positions.json")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    # 48 not 16: 4B leaves ~15GB of a 24GB card free, and generation is
    # bandwidth bound -- a bigger batch amortises the weight read across
    # more sequences, which is the actual bottleneck.
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                        help="vllm uses continuous batching (much faster, "
                             "but installs its own pinned torch)")
    # 1024, not 256: Qwen3 writes markdown reasoning even with thinking off,
    # and at 256 several answers were cut mid-sentence -- which scores as
    # ignorance. Truncation is reported so this stays visible.
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--think", action="store_true",
                        help="enable thinking (off by default: it never terminated)")
    parser.add_argument("--model", default=MODEL_ID,
                        help="HF model id, e.g. Qwen/Qwen3-4B")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.positions) as f:
        records = json.load(f)

    items = moves_probe(args.n, rng) + board_probe(records, args.n, rng)
    print(f"{len(items)} probes")

    print(f"model: {args.model}")
    model, tok = load(args.model, backend=args.backend)
    texts, lengths = generate(
        model, tok, [i["prompt"] for i in items],
        max_new_tokens=args.max_new_tokens, think=args.think,
        batch_size=args.batch_size,
    )

    for item, text, length in zip(items, texts, lengths):
        # Recorded per item so a results file is self-describing. Comparing a
        # think=True run against a think=False one is meaningless, and without
        # this there is no way to tell them apart after the fact.
        item["model"] = args.model
        item["think"] = args.think
        item["budget"] = args.max_new_tokens
        item["said"] = boxed(text).strip()
        item["score"] = grade(item, text)
        item["completion"] = text
        item["tokens"] = length
        # Without this, "did not know" and "ran out of budget" are the same
        # number -- and a bigger model that reasons more looks *worse*.
        item["truncated"] = length >= args.max_new_tokens

    for kind in ("moves", "board"):
        scored = [i for i in items if i["kind"] == kind]
        if scored:
            correct = sum(i["score"] for i in scored)
            cut = sum(i["truncated"] for i in scored)
            print(f"\n{kind}: {correct:.0f}/{len(scored)} = "
                  f"{correct / len(scored):.0%}   "
                  f"truncated {cut}/{len(scored)}   "
                  f"mean {sum(i['tokens'] for i in scored) / len(scored):.0f} tokens")
            for item in scored[:4]:
                print(f"   want {str(item['answer'])[:48]:50s} got {item['said'][:40]!r}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(items, f, indent=1)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
