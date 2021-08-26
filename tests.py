from rl_utils import VPPEnv, MarkovianVPPEnv, timestamps_headers
import OnlineHeuristic
from tabulate import tabulate
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from garage.tf.baselines import ContinuousMLPBaseline
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


TIMESTEP_IN_A_DAY = 96

########################################################################################################################


def check_env(num_episodes=1, gurobi_models_dir=None, instances_indexes=[0]):
    """
    Simple test function to check that the environment is working properly.
    :param num_episodes: int; the number of episodes to run.
    :param gurobi_models_dir: string; if specified, the gurobi models are saved in this directory.
    :param instances_indexes: list of int; the indexes of the instances to be considered.
    :return:
    """

    random.seed(0)

    # Load predictions, shifts and prices
    predictions = pd.read_csv('data/instancesPredictionsNew.csv')
    predictions = predictions.iloc[instances_indexes]
    shift = np.load('data/optShift.npy')
    cGrid = np.load('data/gmePrices.npy')

    # Create the environment and a garage wrapper for Gym environments
    env = VPPEnv(predictions=predictions,
                 shift=shift,
                 cGrid=cGrid,
                 noise_std_dev=0.02,
                 savepath=os.path.join(gurobi_models_dir, 'env'))
    env = GymEnv(env)

    # Timestamps headers for visualization
    timestamps = timestamps_headers(env.n)
    # Run the episodes
    for i_episode in range(num_episodes):

        print(f'Episode: {i_episode+1}/{num_episodes}')

        # Reset the environment and get the instance index
        env.reset()

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

        results = OnlineHeuristic.heur(pRenPV=np.expand_dims(env.pRenPVreal, axis=0),
                                       tot_cons=np.expand_dims(env.tot_cons_real, axis=0),
                                       display=False)
        real_cost = results['real cost']

        assert_almost_equal(-step.reward,
                            real_cost,
                            decimal=10), "Cost computed by heur() method and reward are different"

        print(f'\nHeuristic cost: {real_cost} | Reward: {-step.reward}\n')

        print('-'*200 + '\n')

    env.close()

########################################################################################################################


def compute_real_cost_with_c_virt(virtual_costs, instances_indexes, num_episodes=200):
    """
    Compute the real cost for a set of instances given the virtual cost associated to the storage.
    :param virtual_costs: string; path from which the virtual costs are loaded.
    :param num_episodes: int; the number of episodes to run.
    :param instances_indexes: list of int; the indexes of the instances to be considered.
    :return:
    """

    assert isinstance(virtual_costs, (str, np.ndarray)), \
        "You must specify either the filename or the values of the virtual costs"

    if isinstance(virtual_costs, str):
        virtual_costs = np.load(virtual_costs, allow_pickle=True)

    random.seed(0)

    # Load predictions, shifts and prices
    predictions = pd.read_csv('data/instancesPredictionsNew.csv')
    predictions = predictions.iloc[instances_indexes]
    shift = np.load('data/optShift.npy')
    cGrid = np.load('data/gmePrices.npy')

    # Create the environment and a garage wrapper for Gym environments
    env = VPPEnv(predictions=predictions,
                 shift=shift,
                 cGrid=cGrid,
                 noise_std_dev=0.02,
                 savepath=None)
    env = GymEnv(env)

    # Timestamps headers for visualization
    timestamps = timestamps_headers(env.n)

    # Keep track of the total cost
    real_costs = []

    # Run the episodes
    for i_episode in range(num_episodes):

        print(f'Episode: {i_episode + 1}/{num_episodes}')

        # Reset the environment and get the instance index
        env.reset()

        results = OnlineHeuristic.heur(pRenPV=np.expand_dims(env.pRenPVreal, axis=0),
                                       tot_cons=np.expand_dims(env.tot_cons_real, axis=0),
                                       virtual_costs=virtual_costs,
                                       display=True)

        real_cost = results['real cost']
        virtual_cost = results['virtual cost']

        real_costs.append(real_cost)

        print(f'\nReal cost: {real_cost} | Virtual cost: {virtual_cost}\n')

        print('-' * 200 + '\n')

    print(f'\nMean cost: {np.mean(real_costs)}\n')

    env.close()

    return results

########################################################################################################################


def check_markovian_env(num_episodes=100):
    """
    Simple test function to check that the environment is working properly
    :return:
    """

    predictions = pd.read_csv('data/instancesPredictionsNew.csv')
    shift = np.load('data/optShift.npy')
    cGrid = np.load('data/gmePrices.npy')

    env = MarkovianVPPEnv(predictions=predictions,
                          shift=shift,
                          cGrid=cGrid)

    env = GymEnv(env)

    timestamps = timestamps_headers(env.n)

    for i_episode in range(num_episodes):

        print(f'Episode: {i_episode}/{num_episodes}')

        obs, info = env.reset()
        instance_idx = env.mr

        done = False

        total_cost = 0
        all_rewards = []

        while not done:
            # env.render(mode='ascii')
            action = env.cGrid.copy()[env.timestep]
            step = env.step(action)
            total_cost -= step.reward
            all_rewards.append(-step.reward)
            done = step.terminal

            if not step.env_info['feasible']:
                break

        if step.env_info['feasible']:
            results = OnlineHeuristic.heur(pRenPV=np.expand_dims(env.pRenPVreal, axis=0),
                                           tot_cons=np.expand_dims(env.tot_cons_real, axis=0),
                                           display=False)

            real_cost = results['real cost']
            all_real_costs = results['all real costs']

            assert_almost_equal(total_cost,
                                real_cost,
                                decimal=10), "Cost computed by heur() method and sum of the rewards are different"
            assert np.array_equal(all_rewards, all_real_costs), "Rewards are not the same"

            print(f'Cumulative reward: {total_cost} | Cost computed by original method: {real_cost}')

    env.close()

########################################################################################################################


# NOTE: set the logdir
@wrap_experiment(log_dir='models/mdp-env_1')
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
        predictions = pd.read_csv('data/instancesPredictionsNew.csv')
        shift = np.load('data/optShift.npy')
        cGrid = np.load('data/gmePrices.npy')

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
        env = MarkovianVPPEnv(predictions=train_predictions,
                     shift=shift,
                     cGrid=cGrid,
                     noise_std_dev=0.02,
                     savepath=None)

        # Garage wrapping of a gym environment
        env = GymEnv(env, max_episode_length=96)

        # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
        policy = GaussianMLPPolicy(env.spec)

        # A value function using a MLP network.
        baseline = ContinuousMLPBaseline(env_spec=env.spec)

        # It's called the "Local" sampler because it runs everything in the same process and thread as where
        # it was called from.
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=96,
                               is_tf_worker=True)

        # Vanilla Policy Gradient
        algo = VPG(env_spec=env.spec,
                   baseline=baseline,
                   policy=policy,
                   sampler=sampler,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.01, ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=num_epochs, batch_size=100 * 96, plot=False)


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
                # env.render(mode='ascii')
                a, agent_info = policy.get_action(last_obs)
                a = agent_info['mean']
                a.dump(os.path.join(log_dir, 'cvirt.npy'))

                '''print('\nAction')
                print(tabulate(np.expand_dims(a, axis=0), headers=timestamps, tablefmt='pretty'))'''

                step = env.step(a)

                total_reward += step.reward

                print(f'\nCost: {-step.reward}')

                if step.terminal or step.timeout:
                    break
                last_obs = step.observation

        print(f'\nMean reward: {-total_reward / num_episodes}')

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
    # check_env(num_episodes=500, gurobi_models_dir='gurobi-models')
    i = 0
    compute_real_cost_with_c_virt(virtual_costs=f'models/single-step-env_{i+1}/cvirt.npy',
                                  instances_indexes=[i],
                                  num_episodes=1)

    '''for i in range(10):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        train_rl_algo(test_split=[i], num_epochs=100)'''

    # check_markovian_env()
    '''for i in range(1, 11):
        test_rl_algo(log_dir=os.path.join('models', f'single-step-env_{i}'),
                     num_episodes=1)'''
