import random
import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
from q_learning import Q_learner, Q_learner_exploration
import matplotlib.pyplot as plt


def M(opponent_epsilon, learner, N=500):
    """
    compute the M value (the average reward) for the learner over N games alternating roles.
    :param opponent_epsilon: opponent epsilon used for the optimal player implementation
    :param learner: Q-learning trained
    :param N: number of games played
    :return: average reward for the learner
    """
    average_reward = 0
    env = TictactoeEnv()
    players = ['X', 'O']

    for i in range(N):
        # initialize game
        env.reset()
        learner.player = players[i % 2]
        opponent = OptimalPlayer(epsilon=opponent_epsilon, player=players[(i + 1) % 2])

        while not env.end:
            if env.current_player == learner.player:  # Q-learning plays
                action = learner.act(env.grid)
                env.step(action)
            else:  # opponent plays
                opponent_action = opponent.act(env.grid)
                env.step(opponent_action)

        average_reward += env.reward(learner.player)

    return average_reward/N


def M_opt(learner, N=500):
    """
    Shorthand for M(0, learner, N)
    """
    return M(0, learner, N=N)


def M_rand(learner, N=500):
    """
    Shorthand for M(1, learner, N)
    """
    return M(1, learner, N=N)


def get_state(byte_array):
    """
    Recover the state (a 3x3 np.array) from a corresponding byte array.
    :byte_array: byte array corresponding to a state
    :return: corresponding state as a 3x3 np.array
    """
    return np.frombuffer(byte_array, dtype=np.float64).reshape(3, 3)


def Q_s_as_matrix(Q_s_dict):
    """
    Convert a given Q(state,-) in dictionary format (as stored in Q_learner) to a matrix format (3x3 grid).
    :Q_s_dict: Q(state,-) in dictionary format
    :return: Q(state,-) in matrix format (3x3 grid)
    """
    Q_s_mat = np.zeros((3, 3))
    for key in Q_s_dict.keys():
        Q_s_mat[int((key - (key % 3)) / 3), key % 3] = Q_s_dict[key]

    return Q_s_mat


def show_Q_s(q_learner, state):
    """
    Plot Q(state,-) as a heatmap for a given state and q_learner.
    """
    assert state.dtype == np.float64  # ensure that state is in the correct format
    Q_s_dict = q_learner.get_Q(state)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    matrix_axes = axes.matshow(Q_s_as_matrix(Q_s_dict))
    figure.colorbar(matrix_axes)

    return
