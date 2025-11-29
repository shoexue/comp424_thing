from agents.agent import Agent
from store import register_agent

import numpy as np
import time
from copy import deepcopy

from helpers import (
    random_move,
    execute_move,
    check_endgame,
    get_valid_moves,
    MoveCoordinates,
)


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Ataxx Student Agent for COMP 424

    Uses:
    - Iterative deepening
    - Minimax with alpha-beta pruning
    - Cheaper heuristic
    - Branching factor control (only explore top-K moves in search)
    - Simple endgame closer heuristic
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "student_agent"

        # Time and search configuration
        self.time_limit = 1.6      # seconds per move (stay < 2.0s)
        self.max_depth_cap = 6     # absolute upper bound on depth
        self.max_branch_inner = 8  # how many best moves to explore at inner nodes

    # ===================== TIME HELPERS =====================

    def _now(self):
        # Use monotonic clock for reliable timing
        return time.monotonic()

    def _time_up(self, start_time):
        return self._now() - start_time >= self.time_limit

    # ===================== PUBLIC ENTRYPOINT =====================

    def step(self, chess_board, player, opponent):
        """
        Called by the simulator for each move.

        chess_board: np.ndarray
        player:      1 or 2 (this agent)
        opponent:    1 or 2 (other player)

        Must return a MoveCoordinates or None.
        """
        start_time = self._now()

        legal_moves = get_valid_moves(chess_board, player)
        if not legal_moves:
            print(f"[StudentAgent] No valid moves for player {player}")
            return None

        # ------------------------------------------------------------------
        # ENDGAME CLOSER:
        # If the opponent has no valid moves and we are ahead or tied,
        # choose a greedy move that:
        #   - maximizes our piece count
        #   - minimizes remaining empty squares
        # ------------------------------------------------------------------
        opp_moves = get_valid_moves(chess_board, opponent)
        my_count = np.count_nonzero(chess_board == player)
        opp_count = np.count_nonzero(chess_board == opponent)

        if len(opp_moves) == 0 and my_count >= opp_count and not self._time_up(start_time):
            print("[StudentAgent] SWITCHING TO GREEDY endgame closer")

            best_move = None
            best_score = -1e9

            for move in legal_moves:
                if self._time_up(start_time):
                    break

                board_copy = deepcopy(chess_board)
                execute_move(board_copy, move, player)

                empties_after = np.count_nonzero(board_copy == 0)
                my_after = np.count_nonzero(board_copy == player)

                # Prefer more of our pieces and fewer empties
                score = my_after * 100 - empties_after * 10

                if score > best_score:
                    best_score = score
                    best_move = move

            if best_move is None:
                best_move = random_move(chess_board, player)

            time_taken = self._now() - start_time
            print(
                f"[StudentAgent] Endgame closer: chosen move with greedy score={best_score:.2f}, "
                f"time={time_taken:.3f}s"
            )
            return best_move
        # ------------------------------------------------------------------

        # Fallback in case search is cut off
        best_move_overall = random_move(chess_board, player)
        best_value_overall = -float("inf")

        depth = 1
        while depth <= self.max_depth_cap:
            if self._time_up(start_time):
                break

            move, value, finished = self._search_depth(
                chess_board, player, opponent, depth, start_time
            )

            if not finished:
                # Time cutoff at this depth → keep previous best
                break

            if move is not None and value is not None and value > best_value_overall:
                best_move_overall = move
                best_value_overall = value

            depth += 1

        time_taken = self._now() - start_time
        print(
            f"[StudentAgent] Player {player} chose move with value={best_value_overall:.2f}, "
            f"final depth={depth-1}, time={time_taken:.3f}s"
        )

        return best_move_overall

    # ===================== SEARCH LAYER =====================

    def _search_depth(self, board, me, opp, depth, start_time):
        """
        Search best move for 'me' at a fixed depth, treating root as maximizing.
        Returns (best_move, best_value, finished_full_depth).
        """
        legal_moves = get_valid_moves(board, me)
        if not legal_moves:
            # No moves at root => passing is forced; treat as finished
            return None, None, True

        best_move = None
        best_value = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        # Move ordering at root: use fast eval
        ordered_moves = self._order_moves_fast(
            board, legal_moves, me, me, opp, maximizing=True
        )

        for move in ordered_moves:
            if self._time_up(start_time):
                return best_move, best_value, False  # time cutoff

            child_board = self._simulate_move(board, move, me)

            value, finished = self._minimax(
                child_board,
                depth - 1,
                alpha,
                beta,
                maximizing=False,
                current_turn=opp,   # opponent moves next
                me=me,
                opp=opp,
                start_time=start_time,
                inner_node=True,
            )

            if not finished:
                return best_move, best_value, False

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_move, best_value, True

    def _minimax(
        self,
        board,
        depth,
        alpha,
        beta,
        maximizing,
        current_turn,
        me,
        opp,
        start_time,
        inner_node,
    ):
        """
        Minimax with alpha-beta pruning.
        Returns (value, finished_full_depth).
        `me` is the player this agent is (fixed perspective).
        `current_turn` is whose move it is at this node.
        """
        # Time check
        if self._time_up(start_time):
            # Value is ignored when finished=False; just bubble up cutoff
            return 0, False

        # Terminal or depth 0: evaluate
        is_endgame, _, _ = check_endgame(board)
        if is_endgame or depth == 0:
            v = self._evaluate_board(board, me, opp)
            return v, True

        legal_moves = get_valid_moves(board, current_turn)

        if not legal_moves:
            # Pass turn: other player moves, depth-1
            next_turn = opp if current_turn == me else me
            return self._minimax(
                board,
                depth - 1,
                alpha,
                beta,
                not maximizing,
                current_turn=next_turn,
                me=me,
                opp=opp,
                start_time=start_time,
                inner_node=inner_node,
            )

        # Decide how many moves to consider at this node
        ordered_moves = self._order_moves_fast(
            board, legal_moves, current_turn, me, opp, maximizing
        )
        if inner_node and depth >= 2 and len(ordered_moves) > self.max_branch_inner:
            moves_to_consider = ordered_moves[: self.max_branch_inner]
        else:
            moves_to_consider = ordered_moves

        if maximizing:
            value = -float("inf")

            for move in moves_to_consider:
                if self._time_up(start_time):
                    return value, False

                child_board = self._simulate_move(board, move, current_turn)
                next_turn = opp if current_turn == me else me

                child_val, finished = self._minimax(
                    child_board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=False,
                    current_turn=next_turn,
                    me=me,
                    opp=opp,
                    start_time=start_time,
                    inner_node=True,
                )

                if not finished:
                    return value, False

                value = max(value, child_val)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break

            return value, True

        else:
            value = float("inf")

            for move in moves_to_consider:
                if self._time_up(start_time):
                    return value, False

                child_board = self._simulate_move(board, move, current_turn)
                next_turn = opp if current_turn == me else me

                child_val, finished = self._minimax(
                    child_board,
                    depth - 1,
                    alpha,
                    beta,
                    maximizing=True,
                    current_turn=next_turn,
                    me=me,
                    opp=opp,
                    start_time=start_time,
                    inner_node=True,
                )

                if not finished:
                    return value, False

                value = min(value, child_val)
                beta = min(beta, value)
                if beta <= alpha:
                    break

            return value, True

    # ===================== MOVE ORDERING (FAST) =====================

    def _order_moves_fast(self, board, moves, current_turn, me, opp, maximizing):
        """
        Move ordering using a cheap heuristic evaluated from **my** perspective (`me`),
        regardless of whose turn it is.

        At max nodes (my turn), we sort descending (best for me).
        At min nodes (opp turn), we sort ascending (best for me last).
        """
        scored = []
        for m in moves:
            child = self._simulate_move(board, m, current_turn)
            # Always evaluate from my perspective (me vs opp)
            val = self._fast_eval(child, me, opp)
            scored.append((val, m))

        scored.sort(key=lambda x: x[0], reverse=maximizing)
        return [m for (_, m) in scored]

    def _fast_eval(self, board, color, opponent):
        my_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        piece_diff = my_count - opp_count

        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        my_corners = sum(1 for (i, j) in corners if board[i, j] == color)
        opp_corners = sum(1 for (i, j) in corners if board[i, j] == opponent)
        corner_diff = my_corners - opp_corners

        return piece_diff + 15.0 * corner_diff

    # ===================== FULL EVALUATION (LEAVES) =====================

    def _evaluate_board(self, board, color, opponent):
        """
        Stage-aware evaluation for Ataxx:

        - Terminal: huge +/- score
        - Piece difference (more important late)
        - Corner difference (always big, bigger late)
        - Mobility (important early/mid)
        - Frontier pieces: punish having many exposed pieces
        """

        # ----- Terminal check -----
        is_endgame, p1_score, p2_score = check_endgame(board)
        if is_endgame:
            if color == 1:
                diff = p1_score - p2_score
            else:
                diff = p2_score - p1_score

            if diff > 0:
                return 10000
            elif diff < 0:
                return -10000
            else:
                return 0

        n = board.shape[0]

        # ----- Game phase: 0 = opening, 1 = endgame -----
        empty_count = np.count_nonzero(board == 0)
        total_squares = board.size
        phase = 1.0 - (empty_count / total_squares)  # early ~0, late ~1

        # ----- Piece difference -----
        my_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        piece_diff = my_count - opp_count

        # ----- Corner difference -----
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        my_corners = sum(1 for (i, j) in corners if board[i, j] == color)
        opp_corners = sum(1 for (i, j) in corners if board[i, j] == opponent)
        corner_diff = my_corners - opp_corners

        # ----- Mobility (only really matters earlier) -----
        my_moves = len(get_valid_moves(board, color))
        opp_moves = len(get_valid_moves(board, opponent))

        # ----- Frontier pieces (pieces touching empty squares) -----
        dirs = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]

        frontier_my = 0
        frontier_opp = 0
        for i in range(n):
            for j in range(n):
                if board[i, j] == color or board[i, j] == opponent:
                    is_frontier = False
                    for di, dj in dirs:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and board[ni, nj] == 0:
                            is_frontier = True
                            break
                    if is_frontier:
                        if board[i, j] == color:
                            frontier_my += 1
                        else:
                            frontier_opp += 1

        # ----- Weights: depend on phase -----
        piece_w      = 0.5 + 1.5 * phase      # 0.5 early, 2.0 late
        corner_w     = 10.0 + 10.0 * phase    # 10 early, 20 late
        my_mob_w     = 0.4 * (1.0 - phase)    # only early/mid
        opp_mob_w    = 1.6 * (1.0 - phase)
        frontier_w   = 0.5 * (1.0 - phase)    # punish exposed pieces early

        value = (
            piece_w    * piece_diff +
            corner_w   * corner_diff +
            my_mob_w   * my_moves -
            opp_mob_w  * opp_moves -
            frontier_w * frontier_my +
            frontier_w * frontier_opp
        )

        return value

    # ===================== UTILITIES =====================

    def _simulate_move(self, board, move, player):
        """
        Returns a new board with `move` applied for `player`.
        """
        new_board = np.copy(board)
        execute_move(new_board, move, player)
        return new_board
