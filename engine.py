"""
This module defines the ground truth the model is graded against
"""

import shutil

import chess
import chess.engine

MATE_SCORE = 10_000
CP_LOSS_CAP = 1_000


class Engine:
    """Owns one Stockfish subprocess. Use as a context manager.

    Fixed depth (not fixed time) with Threads=1 keeps evaluations
    deterministic, so a reward computed on this laptop is reproducible on a
    rented GPU tomorrow. Time-based limits would make runs incomparable.
    """

    def __init__(self, path=None, depth=12, threads=1, hash_mb=128):
        path = path or shutil.which("stockfish")
        if path is None:
            raise RuntimeError(
                "stockfish not found on PATH (try: brew install stockfish)"
            )
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.limit = chess.engine.Limit(depth=depth)
        self._best_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.engine.quit()

    def top_moves(self, board, k=1):
        """[(move, cp), ...] best-first. cp is from the side-to-move's POV.
        might returns fewer than k entries when the position has fewer legal moves.
        """
        infos = self.engine.analyse(board, self.limit, multipv=k)
        return [
            (
                info["pv"][0],
                info["score"].pov(board.turn).score(mate_score=MATE_SCORE),
            )
            for info in infos
        ]

    def best(self, board):
        """(move, cp) for the engine's preferred move, cached by FEN so grpo rollout is faster
        """
        key = board.fen()
        if key not in self._best_cache:
            self._best_cache[key] = self.top_moves(board, k=1)[0]
        return self._best_cache[key]

    def cp_loss(self, board, move):
        """Centipawns given up by `move` versus the engine's best. Always >= 0.

        `board` is left unmodified.
        """
        mover = board.turn
        _, best_cp = self.best(board)

        board.push(move)
        try:
            # Asking Stockfish to analyse a finished position can return
            # garbage or kill the process -- and the position after the single
            # best move in a mate-in-1 is exactly that. Handle it directly.
            if board.is_checkmate():
                after_cp = MATE_SCORE
            elif board.is_stalemate() or board.is_insufficient_material():
                after_cp = 0
            else:
                info = self.engine.analyse(board, self.limit)
                # pov(mover) pins the sign to the player who made the move.
                # Without it the score would be from the opponent's side and
                # the reward would prefer blunders.
                after_cp = info["score"].pov(mover).score(mate_score=MATE_SCORE)
        finally:
            board.pop()

        return min(max(best_cp - after_cp, 0), CP_LOSS_CAP)
