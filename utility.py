import random
import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
from q_learning import Q_learner, Q_learner_exploration


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

