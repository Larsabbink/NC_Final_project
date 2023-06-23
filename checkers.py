import re
import os

class CheckersGame:
    def __init__(self):
        self.board = self.create_board()
        self.current_player = 'W'  # Starting player is white
        self.game_over = False
        self.draw = False
        self.winner = None
        self.error_message = ""
        self.piece = None
        self.moves_without_capture = 0

    def create_board(self):
        board = [[' ']*8 for _ in range(8)]  # 8x8 game board

        for row in range(3):
            for col in range(8):
                if (row + col) % 2 != 1:
                    board[row][col] = 'W'

        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 != 1:
                    board[row][col] = 'B'
 
        return board

    def print_board(self):
        print('   A B C D E F G H')
        print('  -----------------')
        for i in range(7, -1, -1):
            print(f'{i + 1} |{"|".join(self.board[i])}|')
            print('  -----------------')
    
    def make_move(self, move):
        start_row, start_col, end_row, end_col = move

        # Perform the move
        self.board[end_row][end_col] = self.board[start_row][start_col]
        self.board[start_row][start_col] = ' '

        # Check for promotion to king
        if end_row == 0 and self.current_player == 'B':
            self.board[end_row][end_col] = 'BK'
        elif end_row == 7 and self.current_player == 'W':
            self.board[end_row][end_col] = 'WK'
        
        # Capture opponent's pieces if it is a capturing move
        if abs(start_row - end_row) == 2:
            self.moves_without_capture = 0
            capture_row = (start_row + end_row) // 2
            capture_col = (start_col + end_col) // 2
            self.board[capture_row][capture_col] = ' '

            # If another capture is possible let the player make another capture
            _, captured = self.get_possible_moves(end_row, end_col)
            if captured:
                self.piece = [end_row, end_col]
            else:
                self.end_turn()

        else:
            self.moves_without_capture += 1
            self.end_turn()

    def end_turn(self):
        # Change player
        self.current_player = 'W' if self.current_player == 'B' else 'B'   

        self.piece = None

        # Check for game over
        allowed_moves = self.find_all_valid_moves()
        if not allowed_moves:
            self.game_over = True
            self.winner = 'W' if self.current_player == 'B' else 'B'

        if self.moves_without_capture == 40:
            self.game_over = True
            self.draw = True

    def find_all_valid_moves(self):
        allowed_moves = []
        captured_moves = []

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == self.current_player or piece == self.current_player + 'K':
                    moves, captured = self.get_possible_moves(row, col)  
                    if moves and captured:
                        captured_moves += moves
                    elif moves:
                        allowed_moves += moves

        if captured_moves:
            return captured_moves
        else:
            return allowed_moves
        
    def get_possible_moves(self, row, col):
        possible_moves = []
        capturing_moves = []
        captured = False
        if self.board[row][col] in ['WK', 'BK']:
            move_directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
        elif self.current_player == 'W':
            move_directions = [(1, -1), (1, 1)]    # Up-left and up-right
        else:
            move_directions = [(-1, -1), (-1, 1)]     # Down-left and down-right
        
        for direction in move_directions:
            dx, dy = direction
            new_row, new_col = row + dx, col + dy

            # Check if the move is within the board boundaries
            if not self.is_within_board(new_row, new_col):
                continue

            if self.board[new_row][new_col] == ' ':
                possible_moves.append((row, col, new_row, new_col))
                continue

            # Check for capturing move
            enemy_color = 'B' if self.current_player == 'W' else 'W'
            capture_row, capture_col = row + 2*dx, col + 2*dy

            if not self.is_within_board(capture_row, capture_col):
                continue
                
            if self.board[new_row][new_col] == enemy_color and self.board[capture_row][capture_col] == ' ':
                capturing_moves.append((row, col, capture_row, capture_col))
                captured = True
                
        if captured:
            return capturing_moves, captured
        else:
            return possible_moves, captured

    def is_within_board(self, row, col):
        return  (0 <= row < 8 and 0 <= col < 8)

    def play(self):
        while not self.game_over:
            if self.error_message != "":
                print(self.error_message)
                self.error_message = ""
            print("Player to move: " + self.current_player + "\n")
            self.print_board()
            allowed_moves = self.find_all_valid_moves()
            print(list(map(self.move_to_string, allowed_moves)))


            # Get input for the move from the current player
            move = self.get_player_move()

            if not move in allowed_moves:
                self.error_message = "Move is not allowed"
            else:
                self.make_move(move)
                # Logic for game termination

                allowed_moves = self.find_all_valid_moves()
                if not allowed_moves:
                    if self.current_player == 'W':
                        print("Black has won")
                    else:
                        print("White has won")
                    break


            # Clear the terminal output
            os.system('cls')

        self.print_board()
        print("Game Over!")


    def get_player_move(self):
        while True:
            move_input = input("Enter your move (e.g., 'A3 B4'): ")

            # Validate the input format
            move_pattern = r'^[A-H][1-8] [A-H][1-8]$'
            move_input = move_input.strip().upper()

            if not re.match(move_pattern, move_input):
                print("Invalid input. Please enter your move in the format 'A3 B4'.")
                continue

            # Convert the input to row and column indices
            start_col = ord(move_input[0]) - ord('A')
            start_row = int(move_input[1]) - 1
            end_col = ord(move_input[3]) - ord('A')
            end_row = int(move_input[4]) - 1

            return (start_row, start_col, end_row, end_col)
            
    def move_to_string(self, move):
        start_row, start_col, end_row, end_col = move
        
        return chr(start_col + ord('A')) + str(start_row + 1) + ' ' + chr(end_col + ord('A')) + str(end_row + 1)


def main():
    game = CheckersGame()
    game.play()

if __name__ == "__main__":
    main()