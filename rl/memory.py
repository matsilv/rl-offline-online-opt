# Author: Mattia Silvestri

"""
    Trajectories memories.
"""

from collections import deque
import random

########################################################################################################################


class ReplayExperienceBuffer:
    """
        Experiance replay buffer as a deque.
    """
    def __init__(self, maxlen):
        """
        :param maxlen: int; maximum length of the deque.
        """
        self._memory = deque(maxlen=maxlen)

    def insert(self, entry):
        """
        Update the buffer adding a new transition sample.
        :param entry: tuple(numpy.array, numpy.array, float, numpy.array, bool);
                      a tuple with a transition sample as (state_t, action_t, reward, state_{t+1}, done).

        :return:
        """
        self._memory.append(entry)

    # FIXME: add docstring.
    def get_random_batch(self, batch_size):
        return random.sample(list(self._memory), batch_size)

    def reset(self):
        """
        Clear the deque.
        :return:
        """
        self._memory.clear()

    def __len__(self):
        return len(self._memory)

