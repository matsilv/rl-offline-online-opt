# Author: Mattia Silvestri

"""
    RL agents.
"""

import numpy as np
import pickle
import os
from rl.utility import calc_qvals
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import tensorflow as tf
from tensorflow.keras.models import load_model

########################################################################################################################


class DRLAgent:
    """
    Abstract class for Deep Reinforcement Learning agent.
    """

    def __init__(self, env, policy, model, baseline, standardize_q_vals):
        """

        :param env: gym.Environment; the agent interacts with this environment.
        :param policy: policy.Policy; policy defined as a probability distribution of actions over states.
        :param model: model.DRLModel; DRL model.
        :param baseline: baselines.Baseline; baseline used to reduce the variance of the Q-values.
        """

        self._env = env
        self._policy = policy
        self._model = model
        self._baseline = baseline
        self._standardize_q_vals = standardize_q_vals

    def _step(self, action):
        """
        Private method to ensure environment input action is given in the proper format to Gym.
        :param action: numpy.array; the action.
        :return:
        """
        if isinstance(self._env.action_space, Discrete):
            assert action.shape == (self._env.action_space.n,)
            assert np.sum(action) == 1

            action = np.argmax(action)

            return self._env.step(action)
        elif isinstance(self._env.action_space, Box):
            assert action.shape == self._env.action_space.shape

            return self._env.step(action)

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: int; number of interactions with the environment for training.
        :param render: bool; True if you want to render the environment while training.
        :param gamma: float; discount factor.
        :param batch_size: int; batch size.
        :param filename: string; file path where to save/load model weights.
        :return:
        """

        raise NotImplementedError()

    def test(self, loadpath, render, num_episodes=1):
        """
        Test the model.
        :param loadpath: string; model weights loadpath.
        :param render: bool; True if you want to visualize the environment, False otherwise.
        :param num_episodes: int; the number of episodes.
        :return:
        """

        # Load model
        self._model = load_model(loadpath)

        # Loop over the number of episodes
        for _ in range(num_episodes):

            # Initialize the environment
            game_over = False
            s_t = self._env.reset()
            score = 0

            # Perform an episode
            while not game_over:

                # Render if required
                if render:
                    self._env.render()

                # Sample an action from policy
                # Add the batch dimension for the NN model
                probs = self._model(np.expand_dims(s_t, axis=0))
                a_t = self._policy.select_action(probs)

                # Perform a step
                s_tp1, r_t, game_over, _ = self._step(a_t)
                s_t = s_tp1
                score += r_t

            print('Score: {}'.format(score))

########################################################################################################################


class OnPolicyAgent(DRLAgent):
    """
    DRL agent which requires on-policy samples.
    """

    def __init__(self, env, policy, model, baseline, standardize_q_vals):
        """

        :param env: environment on which to train the agent; as Gym environment
        :param policy: policy defined as a probability distribution of actions over states; as policy.Policy
        :param model: DRL model; as models.DRLModel
        :param baseline: baselines.Baseline; baseline used to reduce the variance of the Q-values.
        """

        super(OnPolicyAgent, self).__init__(env, policy, model, baseline, standardize_q_vals)

    def train(self, num_steps, render, gamma, batch_size, filename):
        """
        Training loop.
        :param num_steps: int; training steps in the environment.
        :param render: bool; True if you want to render the environment while training.
        :param gamma: float; discount factor.
        :param batch_size: int; batch size.
        :param filename: string; file path where model weights will be saved.
        :return:
        """

        # Training steps
        steps = 0

        # Sampled trajectory variables
        actions = list()
        states = list()
        q_vals = list()

        score = 0
        num_episodes = 0

        # Keep track of the history
        history = dict()
        history['Number of episodes'] = list()
        history['Steps'] = list()
        history['Score'] = list()
        history['Losses'] = list()
        history['Avg score'] = list()

        while steps < num_steps:

            # Initialize the environment
            game_over = False
            s_t = self._env.reset()

            # Reset current episode states, actions and rewards
            current_states = list()
            current_actions = list()
            current_rewards = list()

            # Keep track of the episode number
            num_episodes += 1

            # Perform an episode
            while not game_over:

                # Render the environment if required
                if render:
                    self._env.render()

                # Sample an action from policy
                # Add the batch dimension for the NN model
                probs = self._model(np.expand_dims(s_t, axis=0))
                action = self._policy.select_action(probs)
                current_actions.append(action)

                # Sample current state, next state and reward
                current_states.append(s_t)
                s_tp1, r_t, game_over, _ = self._step(action)
                current_rewards.append(r_t)
                s_t = s_tp1

                # Increase the score and the steps counter
                score += r_t
                steps += 1

            # Compute the Q-values
            current_q_vals = calc_qvals(current_rewards,
                                        gamma=gamma,
                                        max_episode_length=self._env.max_episode_length)

            # Keep track of trajectories
            states = states + current_states
            actions = actions + current_actions
            q_vals.append(current_q_vals)

            # Training step
            if len(states) >= batch_size:
                # Convert trajectories from list to array
                states = np.asarray(states)
                actions = np.asarray(actions)
                q_vals = np.asarray(q_vals)

                if self._standardize_q_vals:
                    mean = np.nanmean(q_vals, axis=0)
                    std = np.nanstd(q_vals, axis=0)
                    q_vals = (q_vals - mean) / (std + 1e-5)

                # Compute advatange
                adv = self._baseline.compute_advantage(states, q_vals)

                # Perform a gradient descent step
                # Convert states, Q-values and advantage to tensor
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                adv = tf.convert_to_tensor(adv, dtype=tf.float32)
                q_vals = tf.convert_to_tensor(q_vals[~np.isnan(q_vals)], dtype=tf.float32)
                loss_dict = self._model.train_step(states, q_vals, adv, actions)

                # Visualization
                print_string = 'Step: {}/{} | Total reward: {:.2f}'.format(steps, num_steps, score)
                print_string += ' | Total number of episodes: {} | Average score: {:.2f}'.format(num_episodes,
                                                                                                 score / num_episodes)

                # Keep track of history
                history['Number of episodes'].append(num_episodes)
                history['Steps'].append(steps)
                history['Score'].append(score)
                history['Avg score'].append(score / num_episodes)
                history['Losses'].append(loss_dict)

                for loss_name, loss_value in loss_dict.items():
                    print_string += ' | {}: {:.5f} '.format(loss_name, loss_value)

                print(print_string + '\n')
                print('-'*len(print_string) + '\n')

                # Clear trajectory variables
                states = list()
                actions = list()
                q_vals = list()

                # Reset score and number of episodes
                score = 0
                num_episodes = 0

        # Save model and history
        if filename is not None:
            model_savepath = os.path.join(filename, 'agent')
            if not os.path.exists(model_savepath):
                os.makedirs(model_savepath)

            self._model.save(model_savepath)

            pickle.dump(history, open(os.path.join(filename, 'history.pkl'), 'wb'))

########################################################################################################################
