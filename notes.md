# notes

Running log of what I tried, why, and what came out. Numbers live in `runs/`.

---

## The idea

Teach a small Qwen3 to pick chess moves with GRPO. Chess is a good RLVR target
because the verifier is free and graded. Stockfish scores every legal move in
centipawns, so you get a dense gradient instead of a right/wrong bit.

Plan: measure the base model, train, measure again.

---

## Prompt format: FEN to movetext

FEN doesn't work. Given `2b2rk1/...` the model wrote "two kings, two rooks, and
a bishop on the first rank." Completely wrong. Worse, it burned the whole token
budget doing it, while ignoring an ASCII board printed right below.

So I switched to movetext (`1. e4 e5 2. Nf3 ...`). The logic: PGN is what the
model actually saw during pretraining. `gpt-3.5-turbo-instruct` reportedly plays
around 1750 Elo when prompted this way.

It reads movetext fine. It still can't use it. Every move gets echoed back
correctly, then it guesses "stalemate?", then it starts re-listing the same
moves until the budget dies. Legible but not comprehensible. There's no board in
its head.


## The parser was inventing data

Two versions of the same bug. Both flattered the model.

First, scanning prose for move-shaped tokens. A truncated ramble mentioning "the
bishop is on a1, b2" reported `Bh1` as a legal move with a real cp_loss. Fixed
with `\boxed{}`, which is the book's ch3 idiom.

Then 61 of 100 answers turned out to be the literal string `MOVE`. The model was
copying the placeholder out of my own prompt. "answered 95%" was really 34 real
attempts, of which 31 were legal.

Lesson: any metric that can be satisfied by echoing the prompt will be.

---

## The thing that changed my mind

Found a tutorial doing this exact experiment on 8B models. It failed. The model
converged on pushing the a-pawn 80% of the time. A-pawn moves are nearly always
legal and rarely catastrophic, so that's the best constant answer if you can't
read the board. Reward went up the whole time.

DeepSeek's own line: "the improvement is attributed to boosting the correct
response from top K rather than the enhancement of fundamental capabilities."

RL amplifies what a model already does sometimes. It can't install a capability.
They also found their 32B RL run landed on par with the base model, and that
distillation worked where RL didn't.

Two things came out of this. I added a move-distribution panel, so a degenerate
policy shows up instead of hiding behind an improving cp_loss. And I built a
capability probe to check the precondition before spending on training.

---

## The probe

50 questions with mechanically checkable answers.

`moves` asks where a lone piece can go on an empty board. Does it know the
rules? `board` asks what's on a5 after some movetext. Can it track state?

`board` is built 50/50 occupied/empty, so always answering "empty" scores about
50%. Anything at or below that is guessing, not tracking.

### Results across model size

| | moves | board | notes |
|---|---|---|---|
| 0.6B | 0% | 40% | names neighbours, no method, 6 tokens per answer |
| 1.7B | 28% | 40% | correct lines plus spurious squares, 415 tokens |
| 4B | 56% | 44% | nails the knight exactly, 593 tokens |

Rules knowledge scales hard. State tracking doesn't move at all.

All three sit at the ~50% guessing baseline on `board`. 4B is 11/21 = 52% once
you drop its truncations. They get there differently though. 0.6B blurts "empty"
in 6 tokens. 4B reasons for 593 and answers "black rook", wrong. Trying and
failing rather than not trying, but the score is the same.

That's the problem. State tracking is the capability the task needs. Knowing how
a knight moves is useless if you don't know where the knights are.

One error survives every scale: `f9`, `g9`, `h9`. Squares that don't exist. All
three compute the eight neighbours of g8 correctly and none of them clip to the
board. They have movement offsets. None of them has edges.

---

## Gotchas worth remembering

Always report truncation next to a score. At 256 tokens 1.7B scored 12%. At 1024
the same model scored 28%. "Didn't know" and "didn't finish" are different
failures and they look identical in a percentage.

Recompute the random and Stockfish references every run. They shift when the
dataset or engine settings change, and a bare cp_loss means nothing on its own.

The Stockfish ceiling isn't 0 (~11 mean, median 0). Evaluating after a move sits
one ply deeper than the root. Harmless for GRPO, since group normalisation
cancels a constant per-position offset, but it matters when reporting.

---

## Open question

Is there a model size where `board` lifts off 50%? If not, scale isn't the answer
for this task, and distillation (book ch8) is the honest next step. That's also
what DeepSeek concluded for small models.
