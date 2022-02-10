# Author Mattia Silvestri

"""
    Tensorflow 2 models for RL algorithms.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from rl.utility import from_tensor_to_numpy

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
        self._actor_mean = Dense(output_dim)
        self._actor_std_dev = Dense(output_dim, activation=tf.math.softplus)

        # Call build method to define the input shape
        self.build((None,) + input_shape)
        self.compute_output_shape(input_shape=(None, ) + input_shape)

        # Define optimizer
        self._policy_optimizer = Adam()

        # Keep track of trainable variables
        self._actor_trainable_vars = list()
        self._actor_trainable_vars += self._model.trainable_variables
        self._actor_trainable_vars += self._actor_mean.trainable_variables
        self._actor_trainable_vars += self._actor_std_dev.trainable_variables

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

    @from_tensor_to_numpy
    @tf.function
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

    @from_tensor_to_numpy
    #@tf.function
    def train_step(self, states, q_vals, adv, actions):
        """
        Compute loss and gradients. Perform a training step.
        :param states: numpy.array; states of sampled trajectories.
        :param q_vals: list of float; expected return computed with Monte Carlo sampling.
        :param adv: numpy.array; the advantage for each action in the sampled trajectories.
        :param actions: numpy.array; actions of sampled trajectories.
        :return: loss: float; policy loss value.
        """

        # Tape the gradient during forward step and loss computation
        with tf.GradientTape() as policy_tape:
            logits = self.call(inputs=states)
            gaussian = tfp.distributions.Normal(loc=logits[0], scale=logits[1])
            log_prob = gaussian.log_prob(actions)
            neg_log_prob = -tf.reduce_sum(log_prob, axis=1)
            policy_loss = tf.reduce_mean(tf.multiply(neg_log_prob, adv))

        # Perform un update step
        for watched_var, trained_var in zip(policy_tape.watched_variables(), self._actor_trainable_vars):
            assert watched_var.ref() == trained_var.ref()
        dloss_policy = policy_tape.gradient(policy_loss, self._actor_trainable_vars)
        self._policy_optimizer.apply_gradients(zip(dloss_policy, self._actor_trainable_vars))

        return {'Policy loss': policy_loss}

########################################################################################################################



