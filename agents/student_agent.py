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
        self.max_branch_inner = 8 # how many best moves to explore at inner nodes

    # ===================== PUBLIC ENTRYPOINT =====================

    def step(self, chess_board, player, opponent):
        """
        Called by the simulator for each move.

        chess_board: np.ndarray
        player:      1 or 2 (this agent)
        opponent:    1 or 2 (other player)

        Must return a MoveCoordinates or None.
        """
        start_time = time.time()

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
        # This helps actually finish the game instead of bouncing forever.
        # ------------------------------------------------------------------
        opp_moves = get_valid_moves(chess_board, opponent)
        my_count = np.count_nonzero(chess_board == player)
        opp_count = np.count_nonzero(chess_board == opponent)

        if len(opp_moves) == 0 and my_count >= opp_count:
            best_move = None
            best_score = -1e9

            for move in legal_moves:
                board_copy = deepcopy(chess_board)
                execute_move(board_copy, move, player)

                empties_after = np.count_nonzero(board_copy == 0)
                my_after = np.count_nonzero(board_copy == player)

                # Prefer more of our pieces and fewer empties
                score = my_after * 100 - empties_after * 10

                if score > best_score:
                    best_score = score
                    best_move = move

            print(
                f"[StudentAgent] Endgame closer: chosen move with greedy score={best_score:.2f}"
            )
            return best_move
        # ------------------------------------------------------------------

        # Fallback in case search is cut off
        best_move_overall = random_move(chess_board, player)
        best_value_overall = -float("inf")

        depth = 1
        while depth <= self.max_depth_cap:
            if time.time() - start_time > self.time_limit:
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

        time_taken = time.time() - start_time
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
            if time.time() - start_time > self.time_limit:
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
        if time.time() - start_time > self.time_limit:
            return 0, False  # value ignored when finished=False

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
                if time.time() - start_time > self.time_limit:
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
                if time.time() - start_time > self.time_limit:
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
        """
        Cheap eval for move ordering:

        - Piece difference
        - Corner difference (big)
        No mobility here to save time.
        """
        my_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        piece_diff = my_count - opp_count

        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        my_corners = sum(1 for (i, j) in corners if board[i, j] == color)
        opp_corners = sum(1 for (i, j) in corners if board[i, j] == opponent)
        corner_diff = my_corners - opp_corners

        return 1.0 * piece_diff + 15.0 * corner_diff



    # ===================== FULL EVALUATION (LEAVES) =====================

    def _evaluate_board(self, board, color, opponent):
        """
        Strong heuristic:

        - Terminal: huge +/- score
        - Piece difference
        - Corner difference (very high weight)
        - Mobility: reward my moves, heavily punish opponent's moves
        """

        # Terminal check
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

        # Piece difference
        my_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        piece_diff = my_count - opp_count

        # Corner difference
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        my_corners = sum(1 for (i, j) in corners if board[i, j] == color)
        opp_corners = sum(1 for (i, j) in corners if board[i, j] == opponent)
        corner_diff = my_corners - opp_corners

        # Mobility
        my_moves = len(get_valid_moves(board, color))
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_my = my_moves
        mobility_opp = opp_moves

        # Combine with **much stronger** weights
        value = (
            1.0 * piece_diff +         # still matters
            15.0 * corner_diff +       # corners are huge
            0.5 * mobility_my -        # nice to have options
            2.0 * mobility_opp         # very bad if opponent has options
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
