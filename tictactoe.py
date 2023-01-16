"""
Author: chenxy(Chen Xiaoyuan)
Blog:   https://chenxiaoyuan.blog.csdn.net/article/details/128688973

Modified based on https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter01/tic_tac_toe.py

rev1 2023-01-14 
    1000 games, TD0Agent    _win: 0.00, TD0Agent    _win: 0.00, draw_rate = 1.00, tCost = 0.56(sec)
    1000 games, RandomAgent _win: 0.00, TD0Agent    _win: 0.85, draw_rate = 0.14, tCost = 0.48(sec)
    1000 games, TD0Agent    _win: 0.97, RandomAgent _win: 0.00, draw_rate = 0.03, tCost = 0.42(sec)
    1000 games, RandomAgent _win: 0.58, RandomAgent _win: 0.30, draw_rate = 0.12, tCost = 0.49(sec)
    1000 games, RandomAgent _win: 0.00, MinimaxAgent_win: 0.80, draw_rate = 0.20, tCost = 0.62(sec)
    1000 games, MinimaxAgent_win: 1.00, RandomAgent _win: 0.00, draw_rate = 0.00, tCost = 0.52(sec)
    1000 games, MinimaxAgent_win: 0.00, MinimaxAgent_win: 0.00, draw_rate = 1.00, tCost = 0.56(sec)
    1000 games, TD0Agent    _win: 0.00, MinimaxAgent_win: 0.00, draw_rate = 1.00, tCost = 0.50(sec)
    1000 games, MinimaxAgent_win: 0.00, TD0Agent    _win: 0.00, draw_rate = 1.00, tCost = 0.65(sec)
rev2 2023-01-14
(1) State.data --> State.board
(2) Add Agent.name
(3) Recovery Agent.set_symbol() implementation. Remove set_symbol() implementation in Agent children
(4) Refine cache_key in MinimaxAgent::minimax_with_memoization()
(5) Correction to MinimaxAgent::__init__()
rev3 2023-01-14
(1) Agent.states --> Agent.state_history
(2) Class structure refinement
"""
import numpy as np
import pickle
import time
import random

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
DEBUG      = 0 # set to 1 to enable debug message printing

class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.board    = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.board):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.board[i, :]))
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.board[:, i]))

        # check diagonals
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.board[i, i]
            reverse_trace += self.board[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.board))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol):
        new_state = State()
        new_state.board = np.copy(self.board)
        new_state.board[i, j] = symbol
        return new_state

    # print the board
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'X'
                elif self.board[i, j] == -1:
                    token = 'O'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')


def get_all_states_impl(current_state, current_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.board[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)
                    if not is_end:
                        get_all_states_impl(new_state, -current_symbol, all_states)


def get_all_states():
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states


# all possible board configurations
all_states = get_all_states()


class Env:
    def __init__(self):
        self.current_state = State()

    def reset(self):
        self.current_state = State()

    def step(self,action):
        """
        Update board state according to action, and generate the reward
        action: [i,i,player_symbol]
        """
        self.current_state = self.current_state.next_state(action[0],action[1],action[2])
        _ = self.current_state.is_end()
        return self.current_state

class Agent:
    def __init__(self,symbol):
        self.state_history = []
        self.estimations   = dict()
        self.symbol        = symbol
        self.greedy        = []
        self.name          = None

        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5        

    def reset(self):
        self.state_history = []

    def set_state(self, state): # seems better to rename to add_state
        self.state_history.append(state)
        self.greedy.append(True)

    def action(self):
        pass
    def save_policy(self):
        pass
    def load_policy(self):
        pass
    
# AI player
class TD0Agent(Agent):
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, symbol, step_size=0.1, epsilon=0.1):
        super().__init__(symbol)
        #self.estimations = dict()
        self.step_size = step_size
        self.epsilon = epsilon
        #self.state_history = []
        self.greedy = []
        #self.symbol = 0
        self.name   = 'TD0Agent'

    def reset(self):
        super().reset()
        #self.state_history = []
        #self.greedy = []

    def set_state(self, state):
        super().set_state(state)
        #self.state_history.append(state)
        #self.greedy.append(True)

    # update value estimation by back-up
    # This back-up is not that back-up. Refer to backup diagram in Sutton-book.
    def value_update(self): 
        state_history = [state.hash() for state in self.state_history]

        for i in reversed(range(len(state_history) - 1)):
            state = state_history[i]
            td_error = self.greedy[i] * (
                self.estimations[state_history[i + 1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def action(self):
        state = self.state_history[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.board[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())

        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# Minimax player
class MinimaxAgent(Agent):
    def __init__(self, symbol):
        super().__init__(symbol)
        # self.state_history = []
        # self.greedy = []
        self.name   = 'MinimaxAgent'
        self.cache  = dict()

    def reset(self):
        super().reset()
        # self.state_history = []
        # self.greedy = []

    def set_state(self, state):
        super().set_state(state)
        # self.state_history.append(state)
        # self.greedy.append(True)

    # choose an action based on the state, using minimax algorithm
    def minimax_with_memoization(self, state, isMax, player, layer):   
        if DEBUG:
            print('{0}Enter minimax(): state={1}, isMax={2}, player={3}'.format(layer*'+',state.board,isMax,player))

        # cache_key = tuple(np.ravel(state.board)) + (player,)
        # state include information about player, so cache_key can be simplified as below
        cache_key = state.hash()
        
        # Memoization can improve time efficiency by more than two order of magnitude.
        if cache_key in self.cache:
            bestMove, bestScore = self.cache[cache_key]
            # print('minimax(): Recover from cache: {0}, {1} '.format(bestMove, bestScore))
            return bestMove, bestScore
        
        bestScore = -1000 if isMax else 1000
        nextPlayer = -1 * player # Note: two player are represented by 1, -1, respectively
        bestMove = [-1,-1]
        
        # game over judge
        # gameOver, winner = gameJudge(board)    
        gameOver = state.is_end()
        
        if gameOver:
            winner = state.winner
            if winner == 0: # DRAW or TIE game           
                bestScore = 0.5
                if DEBUG:
                    print('{0}GameOver: winner={1}, bestMove={2}, bestScore={3}'.format(layer*'+',winner,bestMove,bestScore))     
                self.cache[cache_key] = (bestMove, bestScore)
                return bestMove,bestScore
            else:
                # If it is the end of game, then it must be the win of the opponent.
                bestScore = (0 if isMax else 1)
                if DEBUG:
                    print('{0}GameOver: winner={1}, bestMove={2}, bestScore={3}'.format(layer*'+',winner,bestMove,bestScore))      
                self.cache[cache_key] = (bestMove, bestScore)
                return bestMove,bestScore

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):  
                if state.board[i][j] == 0:
                    new_state = state.next_state(i, j, player)
                    move, score = self.minimax_with_memoization(new_state, (not isMax), nextPlayer, layer+1)
                    if isMax:
                        if score > bestScore:
                            bestScore = score
                            bestMove  = [i,j]            
                    else:
                        if score < bestScore:
                            bestScore = score        
                            bestMove  = [i,j]              
        if DEBUG:                
            print('{0}Exit minimax(): bestMove={1}, bestScore={2}'.format(layer*'+',bestMove,bestScore))                    
        self.cache[cache_key] = (bestMove, bestScore)
        return bestMove, bestScore        

    def action(self):
        state  = self.state_history[-1]
        layer  = int(np.sum(np.abs(state.board))) # layer is only for debug.
        bestMove, bestScore = self.minimax_with_memoization(state, True, self.symbol, layer)
        action = bestMove.copy()
        action.append(self.symbol)
        if DEBUG:                
            print("bestMove = {0}, bestScore = {1}".format(bestMove, bestScore))
            print('MinimaxAgent::action(): state.board = {0}; action={1}'.format(np.ravel(state.board), action))
        return action
        
# human interface
# input a number pair to put a chessman, example for 3x3 tictactoe.
# | (0,0) | (0,1) | (0,2) |
# | (1,0) | (1,1) | (1,2) |
# | (2,0) | (2,1) | (2,2) |
class HumanPlayer(Agent):
    def __init__(self, symbol):    
        super().__init__(symbol)
        # self.symbol = None
        # self.state  = None
        self.name   = 'HumanPlayer'

    def reset(self):
        super().reset()

    def set_state(self, state):
        super().set_state(state)

    # def set_symbol(self, symbol):
    #     self.symbol = symbol

    def action(self):
        state = self.state_history[-1]
        state.print_state()
        # loop until human make a legal move
        while True:
            try:
                move = input("Enter box location to make your move in format of [i,j], 'q' to abort : ")        
                if move == 'q':
                    return -1,-1,-1
                c1, c2 = move.split(',')
                if not c1.isdigit() or not c2.isdigit():
                    print("Please enter valid move [i,j], each number between 0 and {0}".format(BOARD_ROWS-1))
                else:
                    i,j = int(c1.strip()),int(c2.strip())
                break
            except:
                print("Please enter valid move [i,j], each number between 0 and {0}".format(BOARD_ROWS-1))
        return [i, j, self.symbol]
            
class RandomAgent(Agent):
    def __init__(self, symbol):
        super().__init__(symbol)
        self.state  = None
        self.name   = 'RandomAgent'

    def reset(self):
        super().reset()

    def set_state(self, state):
        super().set_state(state)

    # choose an action from the available empty positions randomly
    def action(self):
        state = self.state_history[-1]
        next_states = []
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.board[i, j] == 0:
                    next_positions.append([i, j])
                    next_states.append(state.next_state(
                        i, j, self.symbol).hash())

        action = next_positions[np.random.randint(len(next_positions))]
        action.append(self.symbol)
        return action

def play_one_game(agent1, agent2, env):
    env.reset()
    agent1.reset()
    agent2.reset()

    while True:
        agent1.set_state(env.current_state)
        action1 = agent1.action()
        if action1[0] == -1:
            return -1
        env.step(action1)
        if env.current_state.end:
            return env.current_state.winner
            
        agent2.set_state(env.current_state)        
        action2 = agent2.action()
        env.step(action2)
        if env.current_state.end:
            return env.current_state.winner
            

def TD0_train(epochs, print_every_n=500):
    player1 = TD0Agent(symbol=1, epsilon=0.01)
    player2 = TD0Agent(symbol=-1,epsilon=0.01)
    env     = Env()
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = play_one_game(player1, player2, env)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.value_update()
        player2.value_update()
        env.reset()
    player1.save_policy()
    player2.save_policy()


def compete(agent1, agent2, num_games):    
    # agent1 play first
    if agent1 == 'TD0Agent':
        player1 = TD0Agent(symbol=1,epsilon=0) 
    elif agent1 == 'MinimaxAgent':
        player1 = MinimaxAgent(symbol=1) 
    elif agent1 == 'RandomAgent':
        player1 = RandomAgent(symbol=1) 
    else:
        print('Invalid agent name {0} for agent1!'.format(agent1))
        return;

    if agent2 == 'TD0Agent':
        player2 = TD0Agent(symbol=-1, epsilon=0) 
    elif agent2 == 'MinimaxAgent':
        player2 = MinimaxAgent(symbol=-1) 
    elif agent2 == 'RandomAgent':
        player2 = RandomAgent(symbol=-1) 
    else:
        print('Invalid agent name {0} for agent2!'.format(agent2))
        return;

    player1_win = 0.0
    player2_win = 0.0
    env = Env()
    player1.load_policy()
    player2.load_policy()
    
    t_start = time.time()           
    for _ in range(num_games):
        winner = play_one_game(player1, player2, env)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        env.reset()
    player1_winrate = player1_win / num_games
    player2_winrate = player2_win / num_games
    draw_rate       = 1 - player1_winrate - player2_winrate
    t_stop  = time.time()
    
    print('{0} games, {1:12}_win: {2:.02f}, {3:12}_win: {4:.02f}, draw_rate = {5:.02f}, tCost = {6:.02f}(sec) '.\
        format(num_games, player1.name, player1_winrate, player2.name, player2_winrate, draw_rate, t_stop-t_start))    
    return

# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def askGameStart():
    # Ask human start a game or not;
    print('Do you want to start a game? Y or y to start; Others to exit');
    inputWord = input().lower();
    if inputWord.startswith('y'):
        startNewGame = True;
    else:
        startNewGame = False;

    return startNewGame
    
def play_human_vs_agent(agent_name):

    while askGameStart():
        # Decide who, either human or AI, play first.
        # 0: computer; 1: human.
        print('Who play first? [0: computer; 1: human; enter: guess first]');
        cmd = input()
        if not cmd.isdigit():
            who_first = random.randint(0,1);
        else:
            if int(cmd)==0:
                who_first = 0
            else:
                who_first = 1    
    
        if who_first == 1: # HumanPlayer first    
            print('You play first!')
            player1 = HumanPlayer(symbol=1 )
            if agent_name == 'TD0Agent':
                player2 = TD0Agent(symbol=-1, epsilon=0) 
            elif agent_name == 'MinimaxAgent':
                player2 = MinimaxAgent(symbol=-1) 
            elif agent_name == 'RandomAgent':
                player2 = RandomAgent(symbol=-1) 
            else:
                print('Invalid agent name {0} for agent!'.format(agent_name))
                return;
            player2.load_policy()
        else:  # Computer first
            print('Computer play first!')
            player2 = HumanPlayer(symbol=-1 )
            if agent_name == 'TD0Agent':
                player1 = TD0Agent(symbol=1, epsilon=0) 
            elif agent_name == 'MinimaxAgent':
                player1 = MinimaxAgent(symbol=1) 
            elif agent_name == 'RandomAgent':
                player1 = RandomAgent(symbol=1) 
            else:
                print('Invalid agent name {0} for agent!'.format(agent_name))
                return;
            player1.load_policy()

        env     = Env()
            
        winner  = play_one_game(player1, player2, env)
        if winner == (2*who_first-1):
            print("You win!")
        elif winner == (1-2*who_first):
            print("You lose!")
        elif winner == 0:
            print("A tie game!")
        else:
            print("Abort the game!")
            
if __name__ == '__main__':
    # TD0_train(int(1e4))

    num_games = 1000
    compete("TD0Agent", "TD0Agent", num_games)
    compete("RandomAgent", "TD0Agent", num_games)
    compete("TD0Agent", "RandomAgent", num_games)
    compete("RandomAgent", "RandomAgent", num_games)    
    compete("RandomAgent", "MinimaxAgent", num_games)    
    compete("MinimaxAgent", "RandomAgent", num_games)    
    compete("MinimaxAgent", "MinimaxAgent", num_games)        
    compete("TD0Agent", "MinimaxAgent", num_games)    
    compete("MinimaxAgent", "TD0Agent", num_games)    

    try:
        play_human_vs_agent('TD0Agent')
    except:
        play_human_vs_agent('TD0Agent')

