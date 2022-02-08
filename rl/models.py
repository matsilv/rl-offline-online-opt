# Author Mattia Silvestri

"""
    Tensorflow 2 models for RL algorithms.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np

########################################################################################################################

DISCRETE_SPACE = "discrete"
CONTINUOUS_SPACE = "continuous"

########################################################################################################################


class DRLModel(tf.keras.Model):
    """
    Deep Reinforcement Learning base class.
    """

    def __init__(self, input_shape, output_dim, hidden_units=[32, 32]):
        """
        :param output_dim: int; output dimension of the neural network, i.e. the actions space.
        :param hidden_units: list of int; units for each hidden layer.
        """

        super(DRLModel, self).__init__()
        self._output_dim = output_dim

        # Define common body
        self._model = Sequential()
        self._model.add(Input(input_shape))
        self._model.add(Dense(units=hidden_units[0], activation='relu'))
        for units in hidden_units[1:]:
            self._model.add(Dense(units=units, activation='relu'))

        # Create the actor
        self._actor = Dense(output_dim)

        # Define optimizer
        self._optimizer = Adam()

    def call(self, inputs):
        """
        Override the call method of tf.keras Model.
        :param inputs: numpy.array or tf.Tensor; the input arrays.
        :return: tf.Tensor; the output logits.
        """
        hidden_state = self._model(inputs)
        logits = self._actor(hidden_state)

        return logits

    def train_step(self, *args, **kwargs):
        """
        A single training step.
        """
        raise NotImplementedError()

########################################################################################################################


class DRLModel(tf.keras.Model):
    """
    Deep Reinforcement Learning base class.
    """

    def __init__(self, input_shape, output_dim, hidden_units=[32, 32]):
        """
        :param output_dim: int; output dimension of the neural network, i.e. the actions space.
        :param hidden_units: list of int; units for each hidden layer.
        """

        super(DRLModel, self).__init__()
        self._output_dim = output_dim

        # Define common body
        self._model = Sequential()
        self._model.add(Input(input_shape))
        self._model.add(Dense(units=hidden_units[0], activation='relu'))
        for units in hidden_units[1:]:
            self._model.add(Dense(units=units, activation='relu'))

        # Create the actor
        self._actor_mean = Dense(output_dim)
        self._actor_std_dev = Dense(output_dim, activation='softplus')

        # Call buil method
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
        hidden_state = self._model(inputs)
        mean = self._actor_mean(hidden_state)
        std_dev = self._actor_std_dev(hidden_state)

        return mean, std_dev

    def train_step(self, *args, **kwargs):
        """
        A single training step.
        """
        raise NotImplementedError()

########################################################################################################################


class PolicyGradient(DRLModel):
    """
        Definition of Policy Gradient RL algorithm.
    """

    def __init__(self, input_shape, output_dim, hidden_units=[32, 32]):
        super(PolicyGradient, self).__init__(input_shape, output_dim, hidden_units)

    def train_step(self, states, q_vals, actions):
        """
        Compute loss and gradients. Perform a training step.
        :param states: numpy.array; states of sampled trajectories.
        :param q_vals: list of float; expected return computed with Monte Carlo sampling.
        :param actions: numpy.array; actions of sampled trajectories.
        :return: loss: float; policy loss value.
        """

        # Compute the baseline as the mean expected cumulative rewards over the batch
        baseline = np.mean(q_vals)

        # Tape the gradient during forward step and loss computation
        with tf.GradientTape(persistent=True) as tape:
            logits = self.call(inputs=states)
            gaussian = tfp.distributions.Normal(loc=logits[0], scale=logits[1])
            neg_log_prob = -tf.reduce_sum(gaussian.log_prob(actions), axis=1)
            policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, q_vals - baseline))

        # Perform un update step
        dloss_policy = tape.gradient(policy_loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(dloss_policy, self.trainable_variables))

        policy_loss = policy_loss.numpy().item()

        return policy_loss

########################################################################################################################



