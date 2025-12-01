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
    - Minimax with alpha-beta pruning
    - Iterative deepening
    - Cheaper heuristic function
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.board_size = 7
        self.time_limit = 1.95
        self.best_move_found = None
        self.best_score_found = float('-inf')
        self.start_time = None
        self.max_depth = 3

    def step(self, board, color, opponent):
        start_time = time.time()
        legal_moves = get_valid_moves(board, color)
        
        if not legal_moves:
            return None
        
        # If few moves or endgame, use deeper search
        if len(legal_moves) <= 5 or np.count_nonzero(board == 0) < 10:
            current_max_depth = 4
        else:
            current_max_depth = self.max_depth
        
        best_move = None
        best_score = float('-inf')
        
        # Iterative deepening with time management
        for depth in range(1, current_max_depth + 1):
            if time.time() - start_time > self.time_limit:
                break
                
            current_best_move = None
            current_best_score = float('-inf')
            
            for move in legal_moves:
                if time.time() - start_time > self.time_limit:
                    break
                    
                simulated_board = deepcopy(board)
                execute_move(simulated_board, move, color)
                
                # Use minimax with opponent's perspective for counter-play
                score = self.minimax(
                    simulated_board, depth-1, False, color, opponent,
                    float('-inf'), float('inf'), start_time
                )
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
            
            # Only update if we completed this depth level
            if time.time() - start_time <= self.time_limit and current_best_move is not None:
                best_move = current_best_move
                best_score = current_best_score
        
        return best_move or legal_moves[0]
    
    def minimax(self, board, depth, maximizing, color, opponent, alpha, beta, start_time):
        # Time check
        if time.time() - start_time > self.time_limit:
            return 0
        
        # Terminal conditions
        if depth == 0:
            return self.advanced_evaluate(board, color, opponent)
        
        current_player = color if maximizing else opponent
        legal_moves = get_valid_moves(board, current_player)
        
        if not legal_moves:
            # If no moves, evaluate current state
            return self.advanced_evaluate(board, color, opponent)
        
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                if time.time() - start_time > self.time_limit:
                    break
                    
                new_board = deepcopy(board)
                execute_move(new_board, move, color)
                eval = self.minimax(new_board, depth-1, False, color, opponent, alpha, beta, start_time)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                if time.time() - start_time > self.time_limit:
                    break
                    
                new_board = deepcopy(board)
                execute_move(new_board, move, opponent)
                eval = self.minimax(new_board, depth-1, True, color, opponent, alpha, beta, start_time)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def advanced_evaluate(self, board, color, opponent):
        """
        Enhanced evaluation function:

        - Piece difference (important later in game)
        - Corner/Edge control (important mid/late game)
        - Mobility (important early [sometimes mid] game)
        - Stability (important early/mid game)
        - Capture potential
        """
        n = board.shape[0]
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        
        # 1. Piece difference (with endgame multiplier)
        empty_cells = np.count_nonzero(board == 0)
        total_cells = n * n
        
        if empty_cells < total_cells * 0.3:  # Endgame - pieces matter more
            score_diff = (player_count - opp_count) * 2
        else:
            score_diff = player_count - opp_count
        
        # 2. Strategic positioning (counters greedy agent)
        corners = [(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)]
        edges = []
        for i in range(1, n-1):
            edges.extend([(0, i), (i, 0), (n-1, i), (i, n-1)])
        
        # Corner value - but less than greedy agent
        corner_score = 0
        for corner in corners:
            if board[corner] == color:
                corner_score += 3  # Less than greedy's +5
            elif board[corner] == opponent:
                corner_score -= 4  # Penalize opponent corners more
        
        # Edge control - something greedy agent ignores
        edge_score = 0
        for edge in edges:
            if board[edge] == color:
                edge_score += 1
            elif board[edge] == opponent:
                edge_score -= 1
        
        # 3. Mobility advantage
        player_moves = len(get_valid_moves(board, color))
        opp_moves = len(get_valid_moves(board, opponent))
        
        if player_moves + opp_moves > 0:
            mobility_score = 2 * (player_moves - opp_moves) / (player_moves + opp_moves) * 10
        else:
            mobility_score = 0
        
        # 4. Stability - pieces that are hard to flip
        stability_score = self.calculate_stability(board, color, opponent, corners, edges)
        
        # 5. Potential captures
        capture_potential = self.evaluate_capture_potential(board, color, opponent)
        
        # Combined score with tuned weights
        total_score = (
            score_diff * 1.0 +
            corner_score * 1.0 +
            edge_score * 0.7 +
            mobility_score * 1.5 +
            stability_score * 1.2 +
            capture_potential * 0.8
        )
        
        return total_score
    
    def calculate_stability(self, board, color, opponent, corners, edges):
        """Calculate how stable pieces are (hard to capture)."""
        n = board.shape[0]
        stable_score = 0
        
        # Corners are very stable
        for corner in corners:
            if board[corner] == color:
                stable_score += 2
        
        # Pieces adjacent to corners are often stable
        for corner in corners:
            i, j = corner
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n and 0 <= nj < n:
                    if board[ni, nj] == color and board[i, j] == color:
                        stable_score += 1
        
        return stable_score
    
    def evaluate_capture_potential(self, board, color, opponent):
        """Evaluate potential for future captures."""
        n = board.shape[0]
        capture_score = 0
        
        # Look for vulnerable opponent pieces
        for i in range(n):
            for j in range(n):
                if board[i, j] == opponent:
                    # Check if this piece can be captured
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            if abs(dx) + abs(dy) in [1, 2]:  # Valid move distances
                                ni, nj = i + dx, j + dy
                                if 0 <= ni < n and 0 <= nj < n and board[ni, nj] == color:
                                    capture_score += 1
        
        return capture_score