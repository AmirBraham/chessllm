"""Charts for a baseline run.

    uv run python plot.py runs/movetext-think.json

Three panels, because three different questions get asked of a run:

  1. how good are the moves    -> cp_loss distribution, model against both
                                  references on the same positions
  2. did the model answer      -> answered / legal / truncated rates
  3. how long did it think     -> completion lengths against the budget

The cp_loss panel is an ECDF rather than a histogram: the arms have very
different shapes (the model has a long blunder tail, Stockfish piles up near
zero) and cumulative curves stay readable where overlapping bars do not. It
also reads medians directly -- follow across from 0.5.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a pod
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e0e0dc"

# Categorical slots 1-3 of the validated palette. Fixed assignment: each arm
# keeps its hue no matter how many arms a run happens to contain.
SERIES = {
    "model": "#2a78d6",
    "random legal move": "#eb6834",
    "stockfish best": "#1baf7a",
}
GOOD_MOVE_CP = 50


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def colour_for(name):
    for key, value in SERIES.items():
        if name.startswith(key):
            return value
    return SERIES["model"]  # anything not a reference is the model arm


def short_name(name):
    if name.startswith("random"):
        return "random"
    if name.startswith("stockfish"):
        return "stockfish"
    return "qwen3"


def ecdf_panel(ax, results):
    """Cumulative distribution of cp_loss, one line per arm."""
    arms = [
        (arm, sorted(r["cp_loss"] for r in arm["rows"] if r["cp_loss"] is not None))
        for arm in results
    ]
    arms = [(arm, losses) for arm, losses in arms if losses]

    for index, (arm, losses) in enumerate(arms):
        colour = colour_for(arm["name"])
        fraction = [(i + 1) / len(losses) for i in range(len(losses))]
        ax.step(losses, fraction, where="post", color=colour, linewidth=2,
                label=short_name(arm["name"]))

        # Direct labels, required by the palette's relief rule (the aqua slot
        # is under 3:1 on this surface). Anchored at a different height per
        # arm rather than at each curve's midpoint: the model and random
        # curves nearly coincide, so midpoint labels land on top of each other.
        target = 0.92 - 0.11 * index
        at = next((i for i, f in enumerate(fraction) if f >= target), len(losses) - 1)
        ax.annotate(
            short_name(arm["name"]),
            xy=(losses[at], fraction[at]), xytext=(8, -3),
            textcoords="offset points", color=INK, fontsize=9,
        )

    ax.axvline(GOOD_MOVE_CP, color=INK_SOFT, linewidth=1, linestyle=":", alpha=0.7)
    ax.annotate(f"good move (<={GOOD_MOVE_CP}cp)",
                xy=(GOOD_MOVE_CP, 0.5), xycoords=ax.get_xaxis_transform(),
                xytext=(6, 0), textcoords="offset points",
                color=INK_SOFT, fontsize=8, rotation=90, va="center")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    ax.set_xlabel("centipawns lost vs Stockfish's best", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("fraction of positions", color=INK_SOFT, fontsize=9)
    ax.set_title("Move quality  (left and up is better)",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_ylim(0, 1.02)
    style(ax)


def rates_panel(ax, model):
    """Answered / legal / truncated, as percentages of all positions."""
    rows = model["rows"]
    n = len(rows)
    names = ["answered", "legal", "truncated"]
    values = [100 * sum(1 for r in rows if r.get(key)) / n for key in names]

    bars = ax.barh(names[::-1], values[::-1], height=0.55,
                   color=SERIES["model"], zorder=3)
    for bar, value in zip(bars, values[::-1]):
        ax.annotate(f"{value:.0f}%", xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)

    ax.set_xlim(0, 105)
    ax.set_xlabel("% of positions", color=INK_SOFT, fontsize=9)
    ax.set_title(f"Did it answer?  (n={n})", color=INK, fontsize=11, loc="left", pad=10)
    style(ax)
    ax.grid(axis="y", visible=False)


def moves_panel(ax, model):
    """Which moves it actually plays -- the degenerate-policy detector.

    A published GRPO run on 8B models converged on pushing the a-pawn over 80%
    of the time: a-pawn moves are nearly always legal and rarely catastrophic,
    so it is the best constant answer when you cannot read the board. Mean
    cp_loss improves while the model learns no chess at all. Concentration
    here is what separates the two.
    """
    moves = [r["move"] for r in model["rows"] if r.get("move")]
    if not moves:
        ax.text(0.5, 0.5, "no legal moves produced", transform=ax.transAxes,
                ha="center", va="center", color=INK_SOFT, fontsize=10)
        ax.set_title("Which moves?", color=INK, fontsize=11, loc="left", pad=10)
        style(ax)
        return

    counts = Counter(moves).most_common(12)
    labels = [move for move, _ in counts][::-1]
    shares = [100 * n / len(moves) for _, n in counts][::-1]

    top_move, top_n = counts[0]
    top_share = 100 * top_n / len(moves)

    ax.barh(labels, shares, height=0.6, color=SERIES["model"], zorder=3)
    for y, share in enumerate(shares):
        ax.annotate(f"{share:.0f}%", xy=(share, y), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    color=INK, fontsize=8)

    verdict = "  <- degenerate" if top_share >= 20 else ""
    ax.set_title(f"Which moves?  top: {top_move} at {top_share:.0f}%{verdict}",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("% of legal answers", color=INK_SOFT, fontsize=9)
    ax.set_xlim(0, max(shares) * 1.25)
    style(ax)
    ax.grid(axis="y", visible=False)


def tokens_panel(ax, model):
    """Completion lengths, with the budget marked."""
    tokens = [r["tokens"] for r in model["rows"] if "tokens" in r]
    if not tokens:
        return
    budget = max(tokens)

    # Every completion truncating gives one identical value, and matplotlib
    # then picks a meaningless axis like 511.6-512.4.
    if min(tokens) == budget:
        ax.set_xlim(0, budget * 1.1)
    ax.hist(tokens, bins=30, color=SERIES["model"], zorder=3)
    ax.axvline(budget, color=INK_SOFT, linewidth=1, linestyle=":", alpha=0.7)
    # Truncated completions pile up exactly at the budget, so the tallest bar
    # is under this line -- label it outside the axes rather than on top of it.
    ax.annotate("budget", xy=(budget, 1.0), xycoords=ax.get_xaxis_transform(),
                xytext=(0, 4), textcoords="offset points",
                color=INK_SOFT, fontsize=8, ha="center")

    ax.set_xlabel("completion length (tokens)", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("positions", color=INK_SOFT, fontsize=9)
    ax.set_title("How long did it think?", color=INK, fontsize=11, loc="left", pad=10)
    style(ax)


def plot_results(path):
    """Render charts for a results JSON. Returns the PNG path."""
    path = Path(path)
    with path.open() as f:
        results = json.load(f)

    model = next(
        (a for a in results if not a["name"].startswith(("random", "stockfish"))),
        None,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
    fig.suptitle(path.stem, color=INK, fontsize=12, x=0.005, ha="left", y=0.995)

    ecdf_panel(axes[0][0], results)
    if model:
        rates_panel(axes[0][1], model)
        tokens_panel(axes[1][0], model)
        moves_panel(axes[1][1], model)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return png


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python plot.py runs/<results>.json")
    print(plot_results(sys.argv[1]))
