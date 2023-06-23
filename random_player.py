import random

class RandomPlayer():
    def predict_move(self, checkers_game, player, opponent):
        legal_moves = checkers_game.find_all_valid_moves()

        return random.choice(legal_moves)
        
        