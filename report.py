"""Render every figure from runs/ to runs/figures/, and print a summary.

    uv run python report.py

The notebook imports these same functions, so there is one implementation and
the figures never drift from what the notebook shows.
"""

import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

from plot import INK, INK_SOFT, SURFACE, plt, style

RUNS = Path("runs")
FIGURES = RUNS / "figures"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SAN = re.compile(r"^(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?)$")
OFF_BOARD = re.compile(r"\b[a-h](\d+)\b")
OUTCOMES = ["legal move", "illegal move", "boxed non-move", "no answer"]


# ---------------------------------------------------------------- loading


def load_probes(runs=RUNS):
    """{(model, mode): items}. Falls back to older filename shapes."""
    out = {}
    for path in sorted(runs.glob("p*-*.json")):
        stem = path.stem
        items = json.load(open(path))
        first = items[0] if items else {}
        model = first.get("model", "").split("/")[-1] or stem.split("-")[1]
        mode = "think" if first.get("think") else "nothink"
        if "think" in stem:  # filename wins when present
            mode = "think" if stem.endswith("-think") else "nothink"
        out[(model, mode)] = items
    return out


def load_baselines(runs=RUNS):
    return {p.stem: json.load(open(p)) for p in sorted(runs.glob("*.json"))
            if not p.stem.startswith(("probe", "p1024", "p500", "p100"))}


def model_arm(results):
    return next(a for a in results
                if not a["name"].startswith(("random", "stockfish")))


def summary(items, kind):
    rows = [i for i in items if i["kind"] == kind]
    if not rows:
        return None
    return {
        "pct": 100 * sum(i["score"] for i in rows) / len(rows),
        "truncated": sum(i.get("truncated", False) for i in rows),
        "n": len(rows),
        "tokens": st.mean(i.get("tokens", 0) for i in rows),
    }


def size_of(model):
    """0.6B -> 0.6, for sorting."""
    match = re.search(r"([\d.]+)B", model)
    return float(match.group(1)) if match else 0.0


# ---------------------------------------------------------------- figures


def fig_probe_scores(probes, path=None):
    """moves and board, per model, grouped by think/nothink."""
    models = sorted({m for m, _ in probes}, key=size_of)
    modes = [x for x in ("nothink", "think") if any(x == m for _, m in probes)]

    board_items = next(iter(probes.values()))
    baseline = 100 * st.mean(
        i["answer"] == "empty" for i in board_items if i["kind"] == "board"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    width = 0.8 / max(len(modes), 1)

    for ax, kind, title in zip(
        axes, ("moves", "board"),
        ("Rules: where can this piece go?", "State: what is on this square?"),
    ):
        for offset, mode in enumerate(modes):
            xs, values = [], []
            for index, model in enumerate(models):
                found = summary(probes.get((model, mode), []), kind)
                if found:
                    xs.append(index + offset * width - 0.4 + width / 2)
                    values.append(found["pct"])
            bars = ax.bar(xs, values, width=width * 0.9, zorder=3,
                          color=PALETTE[offset], label=mode)
            for bar, value in zip(bars, values):
                ax.annotate(f"{value:.0f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", color=INK, fontsize=8)

        if kind == "board":
            ax.axhline(baseline, color=INK_SOFT, linestyle=":", linewidth=1, zorder=4)
            ax.annotate(f'always "empty" ({baseline:.0f}%)',
                        xy=(len(models) - 0.6, baseline), xytext=(0, 5),
                        textcoords="offset points", ha="right",
                        color=INK_SOFT, fontsize=8)

        ax.set_xticks(range(len(models)), models)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% correct", color=INK_SOFT, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
        if len(modes) > 1:
            ax.legend(frameon=False, fontsize=9, labelcolor=INK)
        style(ax)
        ax.grid(axis="x", visible=False)

    fig.tight_layout()
    return _save(fig, path)


def fig_attempt_rate(probes, path=None):
    """How often it names a piece rather than saying "empty"."""
    keys = sorted(probes, key=lambda k: (size_of(k[0]), k[1]))
    labels, said_empty, attempted = [], [], []
    for model, mode in keys:
        board = [i for i in probes[(model, mode)] if i["kind"] == "board"]
        if not board:
            continue
        n_empty = sum("empty" in i["said"].lower() for i in board)
        labels.append(f"{model} {mode}")
        said_empty.append(100 * n_empty / len(board))
        attempted.append(100 - said_empty[-1])

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1.6), facecolor=SURFACE)
    y = range(len(labels))
    ax.barh(y, said_empty, height=0.55, color=PALETTE[0], zorder=3,
            label='said "empty"')
    ax.barh(y, attempted, height=0.55, left=said_empty, color=PALETTE[1],
            zorder=3, label="named a piece", edgecolor=SURFACE, linewidth=2)
    for i, (a, b) in enumerate(zip(said_empty, attempted)):
        for value, left in ((a, 0), (b, a)):
            if value > 8:
                ax.annotate(f"{value:.0f}%", xy=(left + value / 2, i), ha="center",
                            va="center", color="white", fontsize=8)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of board questions", color=INK_SOFT, fontsize=9)
    ax.set_title("Does it even attempt the board question?", color=INK,
                 fontsize=11, loc="left", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def outcome_counts(results):
    rows = model_arm(results)["rows"]
    legal = sum(r["legal"] for r in rows)
    illegal = sum(1 for r in rows if r["answered"] and not r["legal"]
                  and SAN.match((r["raw"] or "").strip()))
    junk = sum(1 for r in rows if r["answered"] and not r["legal"]
               and not SAN.match((r["raw"] or "").strip()))
    return [legal, illegal, junk, len(rows) - legal - illegal - junk]


def fig_baseline_outcomes(baselines, path=None):
    names = list(baselines)
    if not names:
        return None
    counts = [outcome_counts(baselines[n]) for n in names]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 1.4), facecolor=SURFACE)
    left = [0] * len(names)
    for i, (outcome, colour) in enumerate(zip(OUTCOMES, PALETTE)):
        values = [c[i] for c in counts]
        ax.barh(names, values, height=0.5, left=left, color=colour, zorder=3,
                label=outcome, edgecolor=SURFACE, linewidth=2)
        for j, value in enumerate(values):
            if value >= 8:
                ax.annotate(str(value), xy=(left[j] + value / 2, j), ha="center",
                            va="center", color="white", fontsize=9)
        left = [l + v for l, v in zip(left, values)]
    ax.set_xlabel("positions", color=INK_SOFT, fontsize=9)
    ax.set_title("Baseline outcomes", color=INK, fontsize=11, loc="left", pad=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=9, labelcolor=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def fig_move_quality(results, name, path=None):
    """cp_loss ECDF against both references."""
    colours = {"random": "#eb6834", "stockfish": "#1baf7a", "qwen3": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    drew = False

    for arm in results:
        losses = sorted(r["cp_loss"] for r in arm["rows"] if r["cp_loss"] is not None)
        if not losses:
            continue
        drew = True
        key = ("random" if arm["name"].startswith("random")
               else "stockfish" if arm["name"].startswith("stockfish") else "qwen3")
        fraction = [(i + 1) / len(losses) for i in range(len(losses))]
        ax.step(losses, fraction, where="post", color=colours[key], linewidth=2,
                label=f"{key} (n={len(losses)})")
        at = next((i for i, f in enumerate(fraction) if f >= 0.6), len(losses) - 1)
        ax.annotate(key, xy=(losses[at], fraction[at]), xytext=(8, -4),
                    textcoords="offset points", color=INK, fontsize=9)

    if not drew:
        plt.close(fig)
        return None

    ax.axvline(50, color=INK_SOFT, linestyle=":", linewidth=1)
    ax.set_xlabel("centipawns lost vs Stockfish's best", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("fraction of positions", color=INK_SOFT, fontsize=9)
    ax.set_title(f"Move quality: {name}", color=INK, fontsize=11, loc="left", pad=10)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    style(ax)
    fig.tight_layout()
    return _save(fig, path)


def fig_move_distribution(results, name, path=None):
    """The degenerate-policy detector."""
    played = [r["move"] for r in model_arm(results)["rows"] if r.get("move")]
    if not played:
        return None

    counts = Counter(played).most_common(12)
    labels = [m for m, _ in counts][::-1]
    shares = [100 * n / len(played) for _, n in counts][::-1]
    top, n_top = counts[0]
    share = 100 * n_top / len(played)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.barh(labels, shares, height=0.6, color=PALETTE[0], zorder=3)
    for y, value in enumerate(shares):
        ax.annotate(f"{value:.0f}%", xy=(value, y), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8)
    ax.set_title(f"{name}: {len(played)} legal, top {top} at {share:.0f}%"
                 f"{'  <- degenerate' if share >= 20 else ''}",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("% of legal answers", color=INK_SOFT, fontsize=9)
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig, path):
    if path is None:
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- summary


def print_summary(probes, baselines):
    if probes:
        print("PROBE")
        for key in sorted(probes, key=lambda k: (size_of(k[0]), k[1])):
            model, mode = key
            parts = []
            for kind in ("moves", "board"):
                found = summary(probes[key], kind)
                if found:
                    parts.append(f"{kind} {found['pct']:5.1f}%"
                                 f" (cut {found['truncated']}/{found['n']},"
                                 f" {found['tokens']:.0f} tok)")
            print(f"  {model:8s} {mode:8s} " + "   ".join(parts))

            moves = [i for i in probes[key] if i["kind"] == "moves"]
            bad = sum(1 for i in moves for m in OFF_BOARD.finditer(i["said"].lower())
                      if not 1 <= int(m.group(1)) <= 8)
            if bad:
                print(f"  {'':8s} {'':8s} off-board squares named: {bad}")

    if baselines:
        print("\nBASELINE")
        for name, results in baselines.items():
            counts = outcome_counts(results)
            print(f"  {name}: " + ", ".join(
                f"{label} {value}" for label, value in zip(OUTCOMES, counts)))
            for arm in results:
                losses = [r["cp_loss"] for r in arm["rows"]
                          if r["cp_loss"] is not None]
                if losses:
                    good = sum(x <= 50 for x in losses) / len(losses)
                    print(f"      {arm['name'][:32]:34s} n={len(losses):3d}"
                          f"  mean {st.mean(losses):5.0f}"
                          f"  median {st.median(losses):5.0f}  good {good:5.1%}")


def main():
    probes, baselines = load_probes(), load_baselines()
    print_summary(probes, baselines)

    written = []
    if probes:
        written += [fig_probe_scores(probes, FIGURES / "probe-scores.png"),
                    fig_attempt_rate(probes, FIGURES / "probe-attempts.png")]
    if baselines:
        written.append(fig_baseline_outcomes(baselines, FIGURES / "baseline-outcomes.png"))
        for name, results in baselines.items():
            written += [fig_move_quality(results, name, FIGURES / f"quality-{name}.png"),
                        fig_move_distribution(results, name, FIGURES / f"moves-{name}.png")]

    written = [w for w in written if w]
    print(f"\nwrote {len(written)} figures to {FIGURES}/")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
