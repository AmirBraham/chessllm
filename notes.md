# notes

Running log of what I tried, why, and what came out. Numbers live in `runs/`.

---

## The idea

Teach a small Qwen3 to pick chess moves with GRPO. Chess is a good RLVR target
because the verifier is free and *graded* — Stockfish scores every legal move in
centipawns, so there's a dense gradient instead of a right/wrong bit.

Plan: measure the base model → train → measure again.

---

## Prompt format: FEN → movetext

**FEN doesn't work.** Given `2b2rk1/...` the model wrote "two kings, two rooks,
and a bishop on the first rank." Completely wrong, and it burned the whole token
budget doing it — while ignoring an ASCII board printed right below.

**So I switched to movetext** (`1. e4 e5 2. Nf3 ...`). Logic: PGN is what the
model actually saw during pretraining, and `gpt-3.5-turbo-instruct` reportedly
plays ~1750 Elo when prompted this way.

**Result: it reads movetext fine but still can't use it.** Echoes every move back
correctly, then guesses "stalemate?", then starts re-listing the same moves until
the budget dies. Legible but not comprehensible — no board in its head.

---

## Thinking mode is dead for this

100% truncated at 512 tokens, 0/100 legal moves. Not a budget problem — 2048 with
FEN truncated too. It loops because it has nothing to reason *with*.

Related: in ch3 of the book, MATH500 gets ~25% overall but **0/7 on Levels 4-5**.
Same loop, just rarer. Qwen3's post-training was math and code, so it learned to
terminate on those — never on chess.

---

## The parser was inventing data

Two separate versions of the same bug, both flattering:

1. Scanning prose for move-shaped tokens meant a truncated ramble mentioning
   "the bishop is on a1, b2" reported `Bh1` as a legal move with a real cp_loss.
   Fixed with `\boxed{}` (the book's ch3 idiom).
2. Then 61/100 answers turned out to be the literal string `MOVE` — the model
   copying the placeholder out of my own prompt. "answered 95%" was really 34
   real attempts, of which 31 were legal.

Lesson: any metric that can be satisfied by echoing the prompt will be.

---

## The thing that changed my mind

Found a tutorial doing this exact experiment on **8B** models. It failed: the
model converged on **pushing the a-pawn 80% of the time**. A-pawn moves are
nearly always legal and rarely catastrophic, so it's the best constant answer if
you can't read the board. Reward went up the whole time.

DeepSeek's own line: *"the improvement is attributed to boosting the correct
response from top K rather than the enhancement of fundamental capabilities."*
**RL amplifies what a model already does sometimes. It can't install a
capability.** They also found their 32B RL run landed on par with the base model,
and that distillation worked where RL didn't.

Two things came out of this:
- added a **move-distribution panel** so a degenerate policy is visible, not
  hidden behind an improving cp_loss
- built a **capability probe** to check the precondition before spending on
  training

---

## The probe

50 questions with mechanically checkable answers:

- `moves` — where can a lone piece go on an empty board? (does it know the rules)
- `board` — after this movetext, what's on a5? (can it track state)

`board` is built 50/50 occupied/empty, so **always answering "empty" scores ~50%**
— anything at or below that is guessing, not tracking.

### Results across model size

| | moves | board | notes |
|---|---|---|---|
| 0.6B | 0% | 40% | names neighbours, invents rank 9, 6 tokens per answer |
| 1.7B | 28% | 40% | correct lines + spurious squares, 415 tokens per answer |
| 4B | ? | ? | |

**Rules knowledge scales. State tracking doesn't.** Both models sit at the
guessing baseline on `board`, they just get there differently — 0.6B blurts
"empty" in 6 tokens, 1.7B reasons for 415 and lands on "empty" anyway.

That's the problem, because **state tracking is the capability the task needs.**
Knowing how a knight moves is useless if you don't know where the knights are.

Recurring error at every size: `g9`, `f9`, `h9` — squares that don't exist. It
computes neighbours correctly and never clips to the board.

---

## Gotchas worth remembering

- Always report **truncation** next to a score. At 256 tokens 1.7B scored 12%;
  at 1024 it scored 28%. Same model. "Didn't know" and "didn't finish" are
  different failures and they look identical in a percentage.
- Recompute the random/Stockfish references **every run**. They shift when the
  dataset or engine settings change, and a bare cp_loss means nothing alone.
- The Stockfish ceiling isn't 0 (~11 mean, median 0). Evaluating after a move is
  one ply deeper than at the root. Harmless for GRPO — group normalisation
  cancels a constant per-position offset — but it matters when reporting.

---

## Open question

Is there a model size where `board` lifts off 40%? If not, scale isn't the
answer for this task and distillation (book ch8) is the honest next step —
which is also what DeepSeek concluded for small models.
