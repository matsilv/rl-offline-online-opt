# Author: Mattia Silvestri

import numpy as np


########################################################################################################################


def calc_qvals(rewards, gamma, max_episode_length):
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

    res = list(reversed(res))
    res = np.asarray(res)

    if len(res) < max_episode_length:
        empty_array = np.empty(max_episode_length - len(res))
        empty_array.fill(np.nan)
        res = np.concatenate((res, empty_array), axis=0)

    return res

########################################################################################################################


def compute_advantage(q_vals):
    """
    # Compute the baseline as the mean expected cumulative rewards over the batch. This method assumes that the state is
    the only timestep (the second axis of the numpy array).
    :param q_vals: numpy.array of shape (batch_size, n_timesteps); the Q-values for all the trajectories in the batch.
    :return: numpy.array of shape (batch_size * n_timesteps); the adantage for each action in the sampeld trajectories.
    """

    baseline = np.nanmean(q_vals, axis=0, keepdims=True)
    adv = q_vals - baseline
    adv = adv.reshape(-1)
    adv = adv[~np.isnan(adv)]

    return adv

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


def from_tensor_to_numpy(train_step):
    """
    A decorator function to convert the action index to a one-hot encoding vector.
    :param select_action: function; the function to decorate.
    :return: function; the decorated function.
    """
    def convert(self, *args):
        loss_dict = train_step(self, *args)
        for loss_name, loss_val in loss_dict.items():
            loss_dict[loss_name] = loss_val.numpy().item()

        return loss_dict

    return convert
