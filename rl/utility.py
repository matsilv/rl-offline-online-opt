# Author: Mattia Silvestri
import numpy as np


########################################################################################################################


def calc_qvals(rewards, gamma):
    """
    Compute discounted rewards-to-go.
    :param rewards: list of float; episode rewards.
    :param gamma: float; discount factor.
    :return: list of float; expected Q-values.
    """
    res = list()
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r += r * gamma
        res.append(sum_r)

    return list(reversed(res))

########################################################################################################################


def from_integer_to_categorical(select_action):
    """
    A decorator function to convert the action index to a one-hot encoding vector.
    :param select_action: function; the function to decorate.
    :return: function; the decorated function.
    """
    def convert(self, *args):
        action_idx = select_action(self, *args)
        action = np.zeros(shape=(self._num_actions, ))
        action[action_idx] = 1

        return action

    return convert

########################################################################################################################
