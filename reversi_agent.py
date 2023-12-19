""" 
Team members (Section 2)
Miss Supithcha    	Jongphoemwatthanaphon  		        6488045
Miss  Sasasuang  	Pattanakitjaroenchai         		6488052
Miss  Nisakorn 	    Ngaosri                             6488226
"""

"""
This module contains agents that play reversi.
Version 3.1
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value
from typing import Tuple, List

import numpy as np

from game import is_terminal, make_move, get_legal_moves

# ============================================================
# Formulation (DO NOT CHANGE)
# ============================================================

def actions(board: np.ndarray, player: int) -> List[Tuple[int, int]]:
    """Return valid actions."""
    return get_legal_moves(board, player)


def transition(board: np.ndarray, player: int, action: Tuple[int, int]) -> np.ndarray:
    """Return a new board if the action is valid, otherwise None."""

    new_board = make_move(board, action, player)
    return new_board


def terminal_test(board: np.ndarray) -> bool:
    return is_terminal(board)

# ============================================================
# Agent Template (DO NOT CHANGE)
# ============================================================

class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self) -> int:
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self) -> Tuple[int, int]:
        """Return move that skips the turn."""
        return (-1, 0)

    @property
    def best_move(self) -> Tuple[int, int]:
        """Return move after the thinking.

        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions) -> Tuple[int, int]:
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)
            p = Process(
                target=self.search,
                args=(
                    board, valid_actions,
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
                self._move = (int(output_move_row.value), int(output_move_column.value))
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = (int(output_move_row.value), int(output_move_column.value))
        return self.best_move

    @abc.abstractmethod
    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.

        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for
            `output_move_row.value` and `output_move_column.value`
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


# ============================================================
# Random Agent (DO NOT CHANGE)
# ============================================================


# class RandomAgent(ReversiAgent):
#     """An agent that move randomly."""

#     def search(
#             self, board, valid_actions,
#             output_move_row, output_move_column):
#         """Set the intended move to the value of output_moves."""
#         # If you want to "simulate a move", you can call the following function:
#         # transition(board, self.player, valid_actions[0])

#         # To prevent your agent to fail silently we should an
#         # explicit trackback printout.
#         try:
#             # while True:
#             #     pass
#             # time.sleep(0.1)
#             randidx = random.randint(0, len(valid_actions) - 1)
#             random_action = valid_actions[randidx]
#             output_move_row.value = random_action[0]
#             output_move_column.value = random_action[1]
#         except Exception as e:
#             print(type(e).__name__, ':', e)
#             print('search() Traceback (most recent call last): ')
#             traceback.print_tb(e.__traceback__)




class NorAgent(ReversiAgent):
    """A minimax agent."""
    DEPTH_LIMIT = 5

    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # valid_actions = actions(board, self.player)
        # print(valid_actions)
        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.
        v = -999999
        # default to first valid action
        output_move_row.value = valid_actions[0][0]
        output_move_column.value = valid_actions[0][1]
        for action in valid_actions:
            new_v = self.min_value(transition(board, self.player, action), depth=1)
            if new_v > v:
                v = new_v
                output_move_row.value = action[0]
                output_move_column.value = action[1]
        return v


    def min_value(self, board: np.ndarray, depth: int) -> float:
        opponent = self.player * -1 # opponent's turn
        if is_terminal(board):
            return self.utility(board)
        if depth >= NorAgent.DEPTH_LIMIT:
            return self.evaluation(board)
        valid_actions = actions(board, opponent)
        if len(valid_actions) == 0:
            return self.max_value(board, depth + 1)  # skip the turn.
        v = 999999
        for action in valid_actions:
            v = min(v, self.max_value(transition(board, opponent, action), depth+1))
        return v

    def max_value(self, board: np.ndarray, depth: int) -> float:
        if is_terminal(board):
            return self.utility(board)
        if depth >= NorAgent.DEPTH_LIMIT:
            return self.evaluation(board)
        valid_actions = actions(board, self.player)
        if len(valid_actions) == 0:
            return self.min_value(board, depth + 1)  # skip the turn.
        v = -999999
        for action in valid_actions:
            v = min(v, self.min_value(transition(board, self.player, action), depth+1))
        return v

    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999
        else:
            return 0

    def evaluation(self, board: np.ndarray) -> float:
        # a dummy evaluation that return diff scores
        return (board == self.player).sum() - (board == (self.player * -1)).sum()


# TODO: Create your own agent

class NewAgent(ReversiAgent):
    """A minimax agent."""
    """ 
    Setting a depth limit of 6 enables our AI to make reasonably fast decisions.
    Furthermore, through experimentation, we've found that a depth limit of 6 consistently outperforms "NorAgent"
    due to its deeper search capabilities, which contribute to a stronger playing advantage.
    """
    DEPTH_LIMIT = 6
    
    def search(
            self, board, valid_actions,
            output_move_row, output_move_column):
        # Initialize alpha and beta used to keep track of the best choices during the search
        alpha = -999999
        beta = 999999

        if len(valid_actions) == 0:
            output_move_row.value = -1
            output_move_column.value = -1
            return  # skip the turn.
        
        v = -999999 # AI's evaluation of the current board state
        
        # default to first valid action
        output_move_row.value = valid_actions[0][0]
        output_move_column.value = valid_actions[0][1]
        for action in valid_actions:
            # transition function to get the next game state, alpha and beta are passed down to help prune the search tree
            new_v = self.min_value(transition(board, self.player, action), depth=1, alpha=alpha, beta=beta) 
            if new_v > v:
                v = new_v
                output_move_row.value = action[0]
                output_move_column.value = action[1]
        return v # Once all valid actions have been evaluated,return best move it found 

    def min_value(self, board, depth, alpha, beta): # opponent's turn
        # Determine the opponent's player identity 
        opponent = self.player * -1  

        # Check if the depth limit has been reached or if the current state is a terminal state
        if depth >= NewAgent.DEPTH_LIMIT or is_terminal(board):
            return self.evaluation(board)
        
        # Get a list of valid actions for the opponent
        valid_actions = actions(board, opponent)

        # If there are no valid actions for the opponent, pass the turn back to the current player
        if not valid_actions:
            return self.max_value(board, depth + 1, alpha, beta)
        
        for action in valid_actions:
            # Recursively evaluate the maximum value after the current player's response
            new_value = self.max_value(transition(board, opponent, action), depth + 1, alpha, beta)
            if new_value < beta: # Update beta with the minimum of the new value and the current beta
                beta = new_value
            if alpha >= beta: # Prune the search if alpha is greater than or equal to beta
                return beta
            
        return beta # Return the best minimum value 

    def max_value(self, board, depth, alpha, beta):
        # Check if the depth limit is reached or if the board represents a terminal state
        if depth >= NewAgent.DEPTH_LIMIT or is_terminal(board):
            return self.evaluation(board)

        # Get a list of valid actions for the current player
        valid_actions = actions(board, self.player)

        # If there are no valid actions, pass the turn to the opponent
        if not valid_actions:
            return self.min_value(board, depth + 1, alpha, beta)

        for action in valid_actions:
            # Recursively evaluate the minimum value after the opponent's response
            new_value = self.min_value(transition(board, self.player, action), depth + 1, alpha, beta)
            if new_value > alpha: # Update alpha with the maximum of the new value and the current alpha
                alpha = new_value
            if alpha >= beta: # Prune the search if alpha is greater than or equal to beta
                return alpha

        return alpha # Return the best maximum value

    def utility(self, board: np.ndarray) -> float:
        if (board == self.player).sum() > (board == (self.player * -1)).sum():
            return 9999 # the agent has more pieces than the opponent
        elif (board == self.player).sum() < (board == (self.player * -1)).sum():
            return -9999 # the opponent has more pieces
        else:
            return 0 # have an equal number of pieces

    # Resourse used: https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf
    def evaluation(self, board: np.ndarray) -> float:
        # Initialize heuristic weights
        corner_weight = 30
        mobility_weight = 5
        stability_weight = 25
        coin_parity_weight = 25
        """
        The values assigned to these weights are based on the research.
        According to An Analysis of Heuristics in Othello research, it reported that the corner heuristic is the most powerful stand-alone heuristic, it guides the game in a direction that enhances the chances of capturing
        corners. "The greater the number of corners captured, the more control a player can exercise over the middle portions of the board, thus flanking a significant portion of the opponent’s coins."
        Moreover, the reason why stability_weight and coin_parity_weight were assigned to equal weight is that we considered maintaining stable positions on the board and maintaining a favorable piece count to be equally important in the overall evaluation function.
        """

        # Calculate the component-wise heuristic values
        coin_parity = self.calculate_coin_parity(board)
        mobility = self.calculate_mobility(board)
        corners = self.calculate_corners(board)
        stability = self.calculate_stability(board)

        # Calculate the overall utility value
        utility_value = (
            corner_weight * corners +
            mobility_weight * mobility +
            stability_weight * stability +
            coin_parity_weight * coin_parity
        )

        return utility_value
    
    def calculate_mobility(self, board: np.ndarray) -> float:
        # Calculate the mobility heuristic value
        # Counting the number of legal moves available: both the max player (self.player) and the min player (self.player * -1)
        max_player_legal_moves = len(self.find_legal_moves(board, self.player))
        min_player_legal_moves = len(self.find_legal_moves(board, self.player * -1))
        
        if (max_player_legal_moves + min_player_legal_moves) == 0: # Check if No Moves are Available
            return 0
        # If there are legal moves available to at least one of the players
        return 100 * (max_player_legal_moves - min_player_legal_moves) / (max_player_legal_moves + min_player_legal_moves)
        """
        If the max player has more, it's a positive score (advantage).
        If the min player has more, it's a negative score (disadvantage).
        If they have the same number, the score is zero.
        """

    def calculate_corners(self, board: np.ndarray) -> float:
        # Calculate the corner heuristic value
        max_player_corners = self.count_corners(board, self.player)
        min_player_corners = self.count_corners(board, self.player * -1)
        
        if (max_player_corners + min_player_corners) == 0:
            return 0
        return 100 * (max_player_corners - min_player_corners) / (max_player_corners + min_player_corners)
        """
        If the max player controls more corners, the score is positive (advantage).
        If the min player controls more corners, the score is negative (disadvantage).
        If they have the same number of corners, the score is zero.
        """

    def calculate_stability(self, board: np.ndarray) -> float:
        # Calculate the stability heuristic value
        max_player_stable, min_player_stable, max_player_semistable, min_player_semistable, max_player_unstable, min_player_unstable = self.classify_stability(board)
        
        if (max_player_stable + max_player_semistable + max_player_unstable + min_player_stable + min_player_semistable + min_player_unstable) == 0:
            return 0
        return 100 * ((max_player_stable + max_player_semistable + max_player_unstable) - (min_player_stable + min_player_semistable + min_player_unstable)) / (max_player_stable + max_player_semistable + max_player_unstable + min_player_stable + min_player_semistable + min_player_unstable)
    
    def find_legal_moves(self, board: np.ndarray, player: int) -> list:
        legal_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(board, row, col, player): # Check if the move is valid for the given player
                    legal_moves.append((row, col))
        return legal_moves

    def count_corners(self, board: np.ndarray, player: int) -> int:
        corner_count = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)] # Define the coordinates of the four corners
        for corner in corners:  # Check each corner for ownership by the player
            if board[corner] == player:
                corner_count += 1
        return corner_count

    def classify_stability(self, board: np.ndarray) -> tuple:
        max_player_stable, min_player_stable = 0, 0
        max_player_semistable, min_player_semistable = 0, 0
        max_player_unstable, min_player_unstable = 0, 0

        # Iterate through the board to classify coins for stability
        for row in range(8):
            for col in range(8):
                if board[row, col] == self.player:
                    classification = self.classify_coin_stability(board, row, col)  
                    # Update counts based on the classification                 
                    if classification == 'stable': 
                        max_player_stable += 1
                    elif classification == 'semistable':
                        max_player_semistable += 1
                    elif classification == 'unstable':
                        max_player_unstable += 1
                elif board[row, col] == -self.player:
                    classification = self.classify_coin_stability(board, row, col) 
                    # Update opponent's counts based on the classification
                    if classification == 'stable': 
                        min_player_stable += 1
                    elif classification == 'semistable':
                        min_player_semistable += 1
                    elif classification == 'unstable':
                        min_player_unstable += 1

        return max_player_stable, min_player_stable, max_player_semistable, min_player_semistable, max_player_unstable, min_player_unstable

    def classify_coin_stability(self, board: np.ndarray, row: int, col: int) -> str:
        if (row == 0 or row == 7) and (col == 0 or col == 7):
            return 'stable'  # Coins in corners are stable.
        elif (row == 0 or row == 7) or (col == 0 or col == 7):
            return 'semistable'  # Coins along the edges are semi-stable.
        else:
            return 'unstable'  # Other coins are unstable.
    
    def is_valid_move(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        # Check if the move is within the board bounds and the target cell is empty.
        if row < 0 or row >= 8 or col < 0 or col >= 8 or board[row, col] != 0:
            return False

        # Check if there is at least one opponent's coin adjacent to the target cell.
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip the current cell.
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and board[r, c] == -player:
                    return True
        return False
    
    def calculate_coin_parity(self, board: np.ndarray) -> float:
        # Count the number of coins owned by the max player (self.player) and min player (-self.player)
        max_player_coins = (board == self.player).sum()
        min_player_coins = (board == -self.player).sum()

        # Calculate the total number of coins on the board
        total_coins = max_player_coins + min_player_coins
        
        # Check if there are no coins on the board to avoid division by zero
        if total_coins == 0:
            return 0
        # Calculate the coin parity heuristic value
        return 100 * (max_player_coins - min_player_coins) / total_coins



    

    