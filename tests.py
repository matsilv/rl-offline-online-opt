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
from garage.experiment.deterministic import set_seed
from garage.envs.normalized_env import NormalizedEnv
import tensorflow as tf
import cloudpickle
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt

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
@wrap_experiment(log_dir='models/tmp-normalized-rew', use_existing_dir=False)
def train_rl_algo(ctxt=None,
                  mdp=False,
                  test_split=0.25,
                  num_epochs=1000,
                  noise_std_dev=0.01):
    """

    :param ctxt: garage.experiment.SnapshotConfig; the snapshot configuration used by Trainer to create the snapshotter.
                                                   If None, it will create one with default settings.
    :param mdp: boolean; True if you want to use the MDP version of the environment.
    :param test_split: float; float or list of int; fraction or indexes of the instances to be used for test.
    :param num_epochs: int; number of training epochs.
    :param noise_std_dev: float; standard deviation for the additive gaussian noise.
    :return:
    """

    # set_seed(1)

    # A trainer provides a default TensorFlow session using python context
    with TFTrainer(snapshot_config=ctxt) as trainer:

        # Load data from file
        predictions = pd.read_csv('data/InstancesPredictionsNewSample.csv')
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

        if mdp:
            max_episode_length = TIMESTEP_IN_A_DAY
            discount = 0.99

            # Create the environment
            env = MarkovianVPPEnv(predictions=train_predictions,
                                  shift=shift,
                                  cGrid=cGrid,
                                  noise_std_dev=noise_std_dev,
                                  savepath=None)

            # Garage wrapping of a gym environment
            env = GymEnv(env, max_episode_length=max_episode_length)

            # Normalize observations
            env = NormalizedEnv(env, normalize_obs=True)
        else:
            max_episode_length = 1
            discount = 0

            # Create the environment
            env = VPPEnv(predictions=train_predictions,
                         shift=shift,
                         cGrid=cGrid,
                         noise_std_dev=noise_std_dev,
                         savepath=None)

            # Garage wrapping of a gym environment
            env = GymEnv(env, max_episode_length=max_episode_length)

            env = NormalizedEnv(env, normalize_reward=True)

        # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
        policy = GaussianMLPPolicy(env.spec)
        obs, _ = env.reset()

        # A value function using a MLP network.
        baseline = ContinuousMLPBaseline(env_spec=env.spec)

        # It's called the "Local" sampler because it runs everything in the same process and thread as where
        # it was called from.
        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=max_episode_length,
                               is_tf_worker=True)

        # Vanilla Policy Gradient
        algo = VPG(env_spec=env.spec,
                   baseline=baseline,
                   policy=policy,
                   sampler=sampler,
                   discount=discount,
                   optimizer_args=dict(learning_rate=0.01, ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=num_epochs, batch_size=100 * max_episode_length, plot=False)

########################################################################################################################


def test_rl_algo(log_dir, test_split, mdp=False, num_episodes=100):
    """
    Test a trained agent.
    :param log_dir: string; path where training information are saved to.
    :param mdp: bool; True if the environment is the MDP version.
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

        # Load data from file
        predictions = pd.read_csv('data/InstancesPredictionsNewSample.csv')
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
        env = VPPEnv(predictions=train_predictions,
                     shift=shift,
                     cGrid=cGrid,
                     noise_std_dev=0,
                     savepath=None)

        policy = algo.policy

        timestamps = timestamps_headers(env.n)
        all_rewards = []

        total_reward = 0
        for episode in range(num_episodes):
            last_obs = env.reset()
            done = False

            episode_reward = 0

            all_actions = []

            # Perform an episode
            while not done:
                # env.render(mode='ascii')
                _, agent_info = policy.get_action(last_obs)
                a = agent_info['mean']
                all_actions.append(np.squeeze(a))

                observations, reward, done, _ = env.step(a)

                total_reward -= reward
                episode_reward -= reward

                if done:
                    break
                last_obs = observations

            print(f'\nTotal reward: {episode_reward}')
            all_rewards.append(episode_reward)

            if mdp:
                all_actions = np.expand_dims(all_actions, axis=0)

            print('\nAction')
            print(tabulate(all_actions, headers=timestamps, tablefmt='pretty'))
            np.save(os.path.join(log_dir, 'cvirt.npy'), np.squeeze(all_actions))

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

    '''sns.set_style('darkgrid')

    instance_idx = 3
    results = \
        compute_real_cost_with_c_virt(virtual_costs='models/optPar333.npy',
                                      instances_indexes=[instance_idx],
                                      num_episodes=1,
                                      noise_std_dev=0,
                                      display=True)

    visualization_df = results['dataframe']

    axes = visualization_df.plot(subplots=True, fontsize=12, figsize=(10, 7))
    plt.xlabel('Timestamp', fontsize=14)

    for axis in axes:
        axis.legend(loc=2, prop={'size': 12})
    plt.plot()
    plt.show()'''

    '''for instance_idx in range(0, 1):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        train_rl_algo(mdp=False, test_split=0.5, num_epochs=100)'''

    test_rl_algo(log_dir=os.path.join('models', 'tmp-normalized-rew'),
                 test_split=0.5,
                 num_episodes=10,
                 mdp=False)