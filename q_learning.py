import random
import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer


class Q_learner:
    """
    Class for implementing the Q-learning-algorithm from Watkins (1989) based on pseudocode
    in Sutton and Barto (2018) 6.5.

    the Q-values are stored as a dictionary indexed on the state (the numpy array as bytes is the key),
    each state entry is then a dictionary with possible actions as keys (int from 0-8) and corresponding Q-values.
    """

    def __init__(self, alpha, gamma, epsilon, Q=None, seed=random.random()):
        random.seed(seed)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player = None
        self.Q = {} if Q is None else Q

    def get_Q(self, state):
        """
        accessor for the Q-values, Q-values are only initialised when needed.
        :param state: state at which we wish to obtain Q-values for the actions
        :return: actions and values arrays
        """
        if state.tobytes() not in self.Q.keys():  # initialize Q-values if it was not done before
            action_value = {}
            for i, pos in enumerate(state.ravel()):  # add only possible actions
                if pos == 0:  # position is empty
                    action_value[i] = 0

            self.Q[state.tobytes()] = action_value

        return self.Q[state.tobytes()]

    def set_Q(self, state, action, value):
        """
        set for the Q-values, Q-values are only initialised when needed.
        :param state: state for which we wish to modify the Q-value
        :param action: action for which we wish to modify the Q-value
        :param value: value to be attributed
        :return: None
        """
        if state.tobytes() not in self.Q.keys():  # initialize Q-values if it was not done before
            action_value = {}
            for i, pos in enumerate(state.ravel()):  # add only possible actions
                if pos == 0:  # position is empty
                    action_value[i] = 0

            self.Q[state.tobytes()] = action_value

        self.Q[state.tobytes()][action] = value

    def act(self, state):
        """
        sample an action based on the greedy policy.
        :param state: state at which we wish to sample actions
        :return: action sampled
        """
        action_value = self.get_Q(state)
        return max(action_value, key=action_value.get)  # return key with max value in dict

    def epsilon_greedy(self, state):
        """
        sample an action based on the epsilon greedy policy.
        :param state: state at which we wish to sample actions
        :return: action sampled
        """
        action_value = self.get_Q(state)
        if random.random() > self.epsilon:
            # return best actions based on Q values
            return max(action_value, key=action_value.get)  # return key with max value in dict
        else:
            return random.choice(list(action_value.keys()))

    def update(self, env, opponent):
        """
        make a move in the TicTacToe Environment based on the Q-learning algorithm, update the Q-values.
        :param env: TicTacToe Environment
        :param opponent: TicTacToe opponent
        :return: boolean flag for end of episode (true if game stopped)
        """
        # play
        state = env.grid.copy()
        action = self.epsilon_greedy(state)
        new_state, end, winner = env.step(action)
        if not end:  # opponent plays
            opponent_action = opponent.act(env.grid)
            new_state, end, winner = env.step(opponent_action)
        reward = env.reward(self.player)

        # update Q-values
        Q_s = self.get_Q(state)
        Q_new_s = self.get_Q(new_state)
        max_value_Q_new_s = max(Q_new_s.values()) if len(Q_new_s.keys()) > 0 else 0
        new_Q_s_a = Q_s[action] + self.alpha * (reward + self.gamma * max_value_Q_new_s - Q_s[action])
        self.set_Q(state, action, new_Q_s_a)

        return end

    def train(self, number_game, opponent_epsilon):
        """
        train the Q-learning against an opponent with given epsilon over a number of games, roles are flipped
        after each game.
        :param number_game: number of games to be played
        :param opponent_epsilon: opponent epsilon used for the optimal player implementation
        """
        env = TictactoeEnv()
        players = ['X', 'O']

        rewards = []

        for i in range(number_game):
            # initialize game
            env.reset()
            self.player = players[i % 2]
            opponent = OptimalPlayer(epsilon=opponent_epsilon, player=players[(i + 1) % 2])

            while not env.end:
                if env.current_player == self.player:  # Q-learning plays
                    self.update(env, opponent)
                else:  # opponent plays
                    opponent_action = opponent.act(env.grid)
                    env.step(opponent_action)

            rewards.append(env.reward(self.player))

        return rewards


class Q_learner_exploration:
    """
    Class for implementing the Q-learning-algorithm from Watkins (1989) based on pseudocode
    in Sutton and Barto (2018) 6.5.

    the Q-values are stored as a dictionary indexed on the state (the numpy array as bytes is the key),
    each state entry is then a dictionary with possible actions as keys (int from 0-8) and corresponding Q-values.

    Modified implementation with decreasing epsilon.
    """

    def __init__(self, alpha, gamma, epsilon_min, epsilon_max, n_star, Q=None, seed=random.random()):
        random.seed(seed)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_star = n_star
        self.player = None
        self.Q = {} if Q is None else Q

    def get_Q(self, state):
        """
        accessor for the Q-values, Q-values are only initialised when needed.
        :param state: state at which we wish to obtain Q-values for the actions
        :return: actions and values arrays
        """
        if state.tobytes() not in self.Q.keys():  # initialize Q-values if it was not done before
            action_value = {}
            for i, pos in enumerate(state.ravel()):  # add only possible actions
                if pos == 0:  # position is empty
                    action_value[i] = 0

            self.Q[state.tobytes()] = action_value

        return self.Q[state.tobytes()]

    def set_Q(self, state, action, value):
        """
        set for the Q-values, Q-values are only initialised when needed.
        :param state: state for which we wish to modify the Q-value
        :param action: action for which we wish to modify the Q-value
        :param value: value to be attributed
        :return: None
        """
        if state.tobytes() not in self.Q.keys():  # initialize Q-values if it was not done before
            action_value = {}
            for i, pos in enumerate(state.ravel()):  # add only possible actions
                if pos == 0:  # position is empty
                    action_value[i] = 0

            self.Q[state.tobytes()] = action_value

        self.Q[state.tobytes()][action] = value

    def act(self, state):
        """
        sample an action based on the greedy policy.
        :param state: state at which we wish to sample actions
        :return: action sampled
        """
        action_value = self.get_Q(state)
        return max(action_value, key=action_value.get)  # return key with max value in dict

    def epsilon_greedy(self, state, iteration):
        """
        sample an action based on the epsilon greedy policy.
        :param state: state at which we wish to sample actions
        :param iteration: episode iterations to decrease exploration level
        :return: index of action sampled
        """
        action_value = self.get_Q(state)
        if random.random() > max(self.epsilon_min, self.epsilon_max*(1-iteration/self.n_star)):
            # return best actions based on Q values
            return max(action_value, key=action_value.get)  # return key with max value in dict
        else:
            return random.choice(list(action_value.keys()))

    def update(self, env, opponent, iteration):
        """
        make a move in the TicTacToe Environment based on the Q-learning algorithm, update the Q-values.
        :param env: TicTacToe Environment
        :param opponent: TicTacToe opponent
        :param iteration: episode iterations to decrease exploration level
        :return: boolean flag for end of episode (true if game stopped)
        """
        # play
        state = env.grid.copy()
        action = self.epsilon_greedy(state, iteration)
        new_state, end, winner = env.step(action)
        if not end:  # opponent plays
            opponent_action = opponent.act(env.grid)
            new_state, end, winner = env.step(opponent_action)
        reward = env.reward(self.player)

        # update Q-values
        Q_s = self.get_Q(state)
        Q_new_s = self.get_Q(new_state)
        max_value_Q_new_s = max(Q_new_s.values()) if len(Q_new_s.keys()) > 0 else 0
        new_Q_s_a = Q_s[action] + self.alpha * (reward + self.gamma * max_value_Q_new_s - Q_s[action])
        self.set_Q(state, action, new_Q_s_a)

        return end

    def train(self, number_game, opponent_epsilon):
        """
        train the Q-learning against an opponent with given epsilon over a number of games, roles are flipped
        after each game.
        :param number_game: number of games to be played
        :param opponent_epsilon: opponent epsilon used for the optimal player implementation
        :return: rewards for each game
        """
        env = TictactoeEnv()
        players = ['X', 'O']

        rewards = []

        for i in range(number_game):
            # initialize game
            env.reset()
            self.player = players[i % 2]
            opponent = OptimalPlayer(epsilon=opponent_epsilon, player=players[(i + 1) % 2])

            while not env.end:
                if env.current_player == self.player:  # Q-learning plays
                    self.update(env, opponent, i)
                else:  # opponent plays
                    opponent_action = opponent.act(env.grid)
                    env.step(opponent_action)

            rewards.append(env.reward(self.player))

        return rewards
