from rl_utils import VPPEnv, MarkovianVPPEnv, timestamps_headers
import OnlineHeuristic
from tabulate import tabulate
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import VPG
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
from garage import wrap_experiment
from garage.envs import GymEnv
import tensorflow as tf
import cloudpickle
import os
import random

########################################################################################################################


def check_env(num_episodes=1, gurobi_models_dir=None):
    """
    Simple test function to check that the environment is working properly.
    :param num_episodes: int; the number of episodes to run.
    :param gurobi_models_dir: string; if specified, the gurobi models are saved in this directory.
    :return:
    """

    random.seed(0)

    # Load predictions, shifts and prices
    predictions = pd.read_csv('instancesPredictions.csv')
    predictions = predictions.iloc[:1]
    shift = np.load('optShift.npy')
    cGrid = np.load('gmePrices.npy')

    # Create the environment and a garage wrapper for Gym environments
    env = VPPEnv(predictions=predictions,
                 shift=shift,
                 cGrid=cGrid,
                 noise_std_dev=0,
                 savepath=os.path.join(gurobi_models_dir, 'env'))
    env = GymEnv(env)

    # Timestamps headers for visualization
    timestamps = timestamps_headers(env.n)

    # Save all the costs and rewards to check that are equals
    all_costs = []
    all_rewards = []

    # Run the episodes
    for i_episode in range(num_episodes):

        print(f'Episode: {i_episode+1}/{num_episodes}')

        # Reset the environment and get the instance index
        env.reset()
        instance_idx = env.mr

        done = False

        while not done:
            print()
            env.render(mode='ascii')
            # To check the correctness of the implementation, we simply set the cGrid as action
            action = env.cGrid.copy()
            print('\nAction')
            print(tabulate(np.expand_dims(action, axis=0), headers=timestamps, tablefmt='pretty'))
            step = env.step(action)

            done = step.terminal

        cost, objList = OnlineHeuristic.heur(pRenPV=np.expand_dims(env.pRenPVreal, axis=0),
                                             tot_cons=np.expand_dims(env.tot_cons_real, axis=0))

        all_rewards.append(-step.reward)
        all_costs.append(cost)

        print(f'Heuristic cost: {cost} | Reward: {-step.reward}')

        print('-'*100 + '\n')

    assert_almost_equal(all_rewards, all_costs, decimal=10), "Cost computed by heur() method and reward are different"

    env.close()

########################################################################################################################


# FIXME: this function has to be fixed considering the new reward function
def check_markovian_env():
    """
    Simple test function to check that the environment is working properly
    :return:
    """
    # NOTE: set the number of episodes
    num_episodes = 1

    predictions = pd.read_csv('instancesPredictions.csv')
    realizations = pd.read_csv('instancesRealizations.csv')
    shift = np.load('optShift.npy')
    cGrid = np.load('gmePrices.npy')

    env = MarkovianVPPEnv(predictions=predictions,
                          realizations=realizations,
                          shift=shift,
                          cGrid=cGrid)

    env = GymEnv(env)

    timestamps = timestamps_headers(env.n)

    for i_episode in range(num_episodes):

        obs, info = env.reset()
        instance_idx = env.mr

        done = False

        total_cost = 0

        while not done:
            env.render(mode='ascii')
            action = env.action_space.sample()
            step = env.step(action)
            total_cost -= step.reward
            done = step.terminal

        cost, objList = heur(instance_idx, 'instancesRealizations.csv')

        assert objList == env.objList, "Step-by-step costs computed by the heuristic and the environment must be " + \
                                       "the same"

    env.close()

########################################################################################################################


# NOTE: set the logdir
@wrap_experiment(log_dir=os.path.join('models', 'gaussian-a2c', 'experiment-2'),
                 archive_launch_repo=False,
                 use_existing_dir=True)
def train_rl_algo(ctxt=None, test_split=0.25, num_epochs=1000):
    """

    :param ctxt: garage.experiment.SnapshotConfig; the snapshot configuration used by Trainer to create the snapshotter.
                                                   If None, it will create one with default settings.
    :param test_split: float; float or list of int; fraction or indexes of the instances to be used for test.
    :param num_epochs: int; number of training epochs.
    :return:
    """

    # A trainer provides a default TensorFlow session using python context
    with TFTrainer(snapshot_config=ctxt) as trainer:

        # Load data from file
        predictions = pd.read_csv('instancesPredictions.csv')
        shift = np.load('optShift.npy')
        cGrid = np.load('gmePrices.npy')

        # Split between training and test
        if isinstance(test_split, float):
            split_index = int(len(predictions) * (1 - test_split))
            train_predictions = predictions[:split_index]
        elif isinstance(test_split, list):
            split_index = test_split
            train_predictions = predictions.iloc[split_index]
        else:
            raise Exception("test_split must be list of int or float")

        # Create the environment
        env = VPPEnv(predictions=train_predictions,
                     shift=shift,
                     cGrid=cGrid,
                     noise_std_dev=0.05,
                     savepath=None)

        # Garage wrapping of a gym environment
        env = GymEnv(env, max_episode_length=1)

        # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
        policy = GaussianMLPPolicy(env.spec)

        # A linear value function (baseline) based on features
        baseline = GaussianMLPBaseline(env_spec=env.spec)

        # It's called the "Local" sampler because it runs everything in the same process and thread as where
        # it was called from.
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=1,
                               is_tf_worker=True)

        # Vanilla Policy Gradient
        algo = VPG(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   sampler=sampler,
                   discount=1,
                   optimizer_args=dict(learning_rate=0.01, ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=num_epochs, batch_size=200, plot=False)

        episode_length = 0
        total_reward = 0
        timestamps = timestamps_headers(env.n)

        for episode in range(100):
            last_obs, _ = env.reset()

            # Perform an episode
            while episode_length < np.inf:
                env.render(mode='ascii')

                a, agent_info = algo.policy.get_action(last_obs)
                a = agent_info['mean']
                print('\nAction')
                print(tabulate(np.expand_dims(a, axis=0), headers=timestamps, tablefmt='pretty'))

                obs, reward, done, info = env.step(a)

                total_reward += reward

                print(f'\nCost: {-reward}')

                episode_length += 1

                if done:
                    break
                last_obs = obs

        print(total_reward / 100)


########################################################################################################################


def test_rl_algo(log_dir, num_episodes=100):
    """
    Test a trained agent.
    :param log_dir: string; path where training information are saved to.
    :param num_episodes: int; number of episodes.
    :return:
    """

    # Create TF1 session and load all the experiments data
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        data = cloudpickle.load(open(os.path.join(log_dir, 'params.pkl'), 'rb'))
        # Get the agent
        algo = data['algo']
        # Create an environment without noise
        env = data['env']

        policy = algo.policy
        policy.reset()

        timestamps = timestamps_headers(env.n)

        total_reward = 0
        for episode in range(num_episodes):
            last_obs, _ = env.reset()
            done = False

            # Perform an episode
            while not done:
                env.render(mode='ascii')

                a, agent_info = policy.get_action(last_obs)
                a = agent_info['mean']
                print('\nAction')
                print(tabulate(np.expand_dims(a, axis=0), headers=timestamps, tablefmt='pretty'))

                step = env.step(a)

                total_reward += step.reward

                print(f'\nCost: {-step.reward}')

                if step.terminal or step.timeout:
                    break
                last_obs = step.obs

        print(f'\nMean reward: {total_reward / num_episodes}')



########################################################################################################################


@wrap_experiment
def resume_experiment(ctxt, saved_dir):
    """Resume a Tensorflow experiment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        saved_dir (str): Path where snapshots are saved.
    """
    with TFTrainer(snapshot_config=ctxt) as trainer:
        trainer.restore(from_dir=saved_dir)
        trainer.resume()


########################################################################################################################


if __name__ == '__main__':
    # check_env(num_episodes=200, gurobi_models_dir='gurobi-models')
    # train_rl_algo(test_split=[0], num_epochs=2)
    # check_markovian_env()
    test_rl_algo(log_dir=os.path.join('models', 'gaussian-a2c', 'experiment-1'))