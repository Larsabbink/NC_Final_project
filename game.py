from checkers import CheckersGame
from model import Model

import numpy as np
import torch
import os

from random_player import RandomPlayer

class Game:
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2
        self.checkers_game = CheckersGame()

    def evaluate_board(self):
        white_score = 0
        black_score = 0

        for row in self.checkers_game.board:
            white_score += row.count('W')
            black_score += row.count('B') * 5
            white_score += row.count('WK')
            black_score += row.count('BK') * 5
        
        if white_score == black_score:
            return (0.5, 0.5)
        elif white_score > black_score:
            return (0.6, 0.4)
        else:
            return (0.4, 0.6)

        
    def play(self):
        while not self.checkers_game.game_over:
            player = self.checkers_game.current_player
            opponent = 'W' if player == 'B' else 'B'

            if player == 'W':
                move = self.player_1.predict_move(self.checkers_game, player, opponent)
            else:
                move = self.player_2.predict_move(self.checkers_game, player, opponent)     

            if move == None:
                print(self.player_1)    
                print(self.player_2)

            self.checkers_game.make_move(move)

        if self.checkers_game.winner == 'W':
            return (1,0)
        elif self.checkers_game.winner == 'B':
            return (0,1)
        else:
            return self.evaluate_board()


# player_1 = Model()
# player_2 = RandomPlayer()
# # player_2 = Model()

# game = Game(player_1, player_2)
# print(game.play())