# Author: Mattia Silvestri

"""
    Baselines used to reduce the variance of the cumulative rewards.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

########################################################################################################################


class Baseline:
    """
    Abstract class for baselines.
    """

    def __init__(self):
        super(Baseline, self).__init__()

    def compute_baseline(self, *args, **kwargs):
        raise NotImplementedError()

    def compute_advantage(self, *args, **kwargs):
        raise NotImplementedError()

########################################################################################################################


class SimpleBaseline(Baseline):
    """
    Simple baseline computed as the mean over the batch.
    """

    def __init__(self):
        super(SimpleBaseline, self).__init__()

    @staticmethod
    def compute_baseline(q_vals):

        baseline = np.nanmean(q_vals, axis=0, keepdims=True)

        return baseline

    def compute_advantage(self, states, q_vals):

        adv = q_vals - self.compute_baseline(q_vals)
        adv = adv.reshape(-1)
        adv = adv[~np.isnan(adv)]

        return adv

########################################################################################################################


class Critic(Baseline, tf.keras.Model):
    """
    Critic Neural Network that approximates the Value-function
    """

    def __init__(self, input_shape, hidden_units=[32, 32]):
        """
        :param hidden_units: list of int; units for each hidden layer.
        """

        super(Critic, self).__init__()

        # Define common body
        self._model = Sequential()
        self._model.add(InputLayer(input_shape))
        for units in hidden_units:
            self._model.add(Dense(units=units, activation='tanh'))
        self._model.add(Dense(1))

        # Call build method to define the input shape
        self.build((None,) + input_shape)
        self.compute_output_shape(input_shape=(None, ) + input_shape)

        # Define optimizer
        self._optimizer = Adam()

    def call(self, inputs):
        """
        Override the call method of tf.keras Model.
        :param inputs: numpy.array or tf.Tensor; the input arrays.
        :return: tf.Tensor; the output logits.
        """

        return self._model(inputs)

    # @tf.function
    def train_step(self, states, q_vals):
        """
        A single training step.
        """

        with tf.GradientTape() as tape:
            value_function = self.call(states)
            loss = tf.reduce_mean(tf.square(q_vals - value_function), axis=0)

        for watched_var, trained_var in zip(tape.watched_variables(), self._model.trainable_variables):
            assert watched_var.ref() == trained_var.ref()
        dloss = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(dloss, self._model.trainable_variables))

        return loss

    def compute_baseline(self, states, q_vals):
        value_function = self.call(states).numpy()
        value_function = np.squeeze(value_function)
        return value_function

    def compute_advantage(self, states, q_vals):

        q_vals = q_vals[~np.isnan(q_vals)]
        adv = q_vals - self.compute_baseline(states, q_vals)

        return adv

