import torch
from torch import nn
import random
import copy

import numpy as np

# softmax = nn.Softmax(dim=0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(5*8*4, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.Softmax(dim=0)
        )
        self.output_index = {
                'NE_1': 0,
                'NE_2': 1,
                'NW_1': 2,
                'NW_2': 3,
                'SE_1': 4,
                'SE_2': 5,
                'SW_1': 6,
                'SW_2': 7
            }

    def forward(self, x):
        x = x.reshape(-1)
        prediction = self.linear_layer(x)
        return prediction
    
    def mutate(self, p):
        def mutate_tensor(tensor):
            for parameter in tensor:
                if random.random() < p:
                    parameter += random.uniform(-0.1, 0.1)
                        

        state_dict = self.state_dict()

        for layer in state_dict:
            tensor = state_dict[layer]
            if len(tensor.shape) == 2:
                for nested_tensor in tensor:
                    mutate_tensor(nested_tensor)
            else:
                mutate_tensor(tensor)

        self.load_state_dict(state_dict)

    def state_to_model_input(self, checkers_game, player, opponent):
        input = np.zeros((5,8,4))

        board = checkers_game.board

        # Loop through the board and fill model_input accordingly
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece == player:
                    input[0][row][col // 2] = 1
                elif piece == opponent:
                    input[1][row][col // 2] = 1
                elif piece == player + 'K':
                    input[2][row][col // 2] = 1
                elif piece == opponent + 'K':
                    input[3][row][col // 2] = 1
                
        # Add the binary representation of the amount of moves since last capture to the input
        binary = format(checkers_game.moves_without_capture, 'b')

        for i, c in enumerate(binary[::-1]):
            input[4][7- (i // 4)][3 - (i % 4)] = int(c)

        # If the player is black rotate first 4 layers such that the network always makes move from the white perspective
        if player == 'B':
            for i in range(4):
                input[i] = np.flip(input[i])

        return torch.from_numpy(input).float()

    def output_to_move(self, checkers_game, output):
        legal_moves = checkers_game.find_all_valid_moves()
        best_move = None
        score_best_move = -1

        for move in legal_moves:
            start_row, start_col, end_row, end_col = move

            direction = ''

            index = start_row * 32 + (start_col // 2) * 8

            if start_row < end_row:
                direction += 'N'
            else:
                direction += 'S'

            if start_col < end_col:
                direction += 'E_'
            else:
                direction += 'W_'

            direction += str(abs(end_col - start_col))
            
            index += self.output_index[direction]
            if output[index] > score_best_move:
                score_best_move = output[index]
                best_move = move

        return best_move

    def predict_move(self, checkers_game, player, opponent):
        model_input = self.state_to_model_input(checkers_game, player, opponent)
        prediction = self.forward(model_input)
        move = self.output_to_move(checkers_game, prediction)

        return move


    
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")


# print("test")
# print(device)
# f1 = open("model_1.txt", "w")
# f1_1 = open("model_1_1.txt", "w")
# f2 = open("model_2.txt", "w")

# model = Model().to(device)
# f1.write(str(model.state_dict()))
# model2 = copy.deepcopy(model)
# model2.mutate(0.05)
# f2.write(str(model2.state_dict()))
# f1_1.write(str(model.state_dict()))
# # print(model)


# X = torch.rand(5, 8, 4, device=device)
# X = np.zeros((5, 8, 4))
# X = torch.from_numpy(X).float()
# # X = X.double()
# prediction_1 = model(X)

# print(prediction_1)
# print(X)
