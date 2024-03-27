import numpy as np

class TicTacToeRL:
    def __init__(self):
        self.q_table = {}  # Q-table to store Q-values for (state, action) pairs

    def get_state_key(self, board):
        # Convert the board state to a hashable key
        return tuple(tuple(row) for row in board)

    def choose_action(self, state, epsilon):
    # Choose an action based on epsilon-greedy strategy
        empty_positions = self.get_empty_positions(state)
        flat_positions = [pos for sublist in empty_positions for pos in sublist]
        
        if np.random.random() < epsilon:
            # Exploration: choose a random valid action
            action = np.random.choice(flat_positions)
        else:
            # Exploitation: choose the action with the highest Q-value
            state_key = self.get_state_key(state)
            if state_key in self.q_table:
                max_q_value_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                action = (max_q_value_action // 3, max_q_value_action % 3)  # Convert 1D index to 2D coordinates
            else:
                action = np.random.choice(flat_positions)
        
        if isinstance(action, int):
            action = (action // 3, action % 3)  # Convert 1D index to 2D coordinates
        
        return action



    def get_empty_positions(self, board):
        # Return a list of empty positions on the board
        empty_positions = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    empty_positions.append((i, j))
        return empty_positions


    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        # Update Q-value of (state, action) pair using Q-learning algorithm
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0}

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}

        max_next_q_value = max(self.q_table[next_state_key].values(), default=0)
        self.q_table[state_key][action] = (1 - alpha) * self.q_table[state_key].get(action, 0) + \
                                           alpha * (reward + gamma * max_next_q_value)

    def play_game(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        # Play a single game using Q-learning
        board = [[' ' for _ in range(3)] for _ in range(3)]
        current_player = 'X'

        while True:
            state = self.get_state_key(board)
            action = self.choose_action(board, epsilon)

            if isinstance(action, tuple):
                row, col = action
            else:
                # If action is not a tuple, set row and col to -1
                row, col = -1, -1

            if row == -1 or col == -1 or board[row][col] != ' ':
                # Invalid move or non-iterable action, skip turn
                continue

            board[row][col] = current_player

            if self.check_win(board, current_player):
                reward = 1  # Win
                next_state = None
                self.update_q_table(state, action, reward, next_state, alpha, gamma)
                break
            elif self.check_draw(board):
                reward = 0.5  # Draw
                next_state = None
                self.update_q_table(state, action, reward, next_state, alpha, gamma)
                break
            else:
                # Switch player
                current_player = 'O' if current_player == 'X' else 'X'

                next_state = self.get_state_key(board)
                reward = 0  # No immediate reward for making a move

                self.update_q_table(state, action, reward, next_state, alpha, gamma)

        return current_player

    def check_win(self, board, player):
        # Check if the player has won the game
        for i in range(3):
            if all(board[i][j] == player for j in range(3)) or \
                    all(board[j][i] == player for j in range(3)):
                return True
        if all(board[i][i] == player for i in range(3)) or \
                all(board[i][2 - i] == player for i in range(3)):
            return True
        return False

    def check_draw(self, board):
        # Check if the game is a draw
        return all(board[i][j] != ' ' for i in range(3) for j in range(3))



if __name__ == "__main__":
    num_episodes = 10000  # Number of training episodes
    epsilon = 0.1  # Epsilon-greedy parameter

    game = TicTacToeRL()

    # Training loop
    for episode in range(num_episodes):
        winner = game.play_game(epsilon=epsilon)
        if episode % 20 == 0:
            print(f"Episode {episode}: Winner - {winner}, Epsilon - {epsilon}")
            epsilon *= 0.99  # Decay epsilon over time

    print("Training complete.")




    # Testing
    print("\nTesting the trained AI:")
    ai_player = 'X'  # The AI player
    human_player = 'O'  # The human player

    while True:
        # Human's move
        print("\nHuman's move:")
        game.display_board()
        while True:
            row = int(input("Enter row number (0-2): "))
            col = int(input("Enter column number (0-2): "))
            if game.make_move(row, col, human_player):
                break
            else:
                print("Invalid move. Try again.")

        # Check if human wins
        if game.check_win(human_player):
            print("Human wins!")
            break
        elif game.check_draw():
            print("It's a draw!")
            break

        # AI's move
        print("\nAI's move:")
        game.display_board()
        row, col = game.choose_action(game.board, epsilon=0)  # Choose the action with no exploration
        game.make_move(row, col, ai_player)

        # Check if AI wins
        if game.check_win(ai_player):
            print("AI wins!")
            break
        elif game.check_draw():
            print("It's a draw!")
            break
