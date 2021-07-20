from rl_utils import VPPEnv, MarkovianVPPEnv, timestamps_headers
from OnlineHeuristic import heur
from tabulate import tabulate
import numpy as np
import pandas as pd
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import VPG
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import TFTrainer
from garage import wrap_experiment
from garage.envs import GymEnv
import tensorflow as tf
import cloudpickle

########################################################################################################################


def check_env():
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

    env = VPPEnv(predictions=predictions,
                 realizations=realizations,
                 shift=shift,
                 cGrid=cGrid)

    env = GymEnv(env)

    timestamps = timestamps_headers(env.n)

    for i_episode in range(num_episodes):

        env.reset()
        instance_idx = env.mr

        done = False

        while not done:
            env.render(mode='ascii')
            action = env.action_space.sample()
            print('\nAction')
            print(tabulate(np.expand_dims(action, axis=0), headers=timestamps, tablefmt='pretty'))
            step = env.step(action)

            cost, _ = heur(instance_idx, 'instancesRealizations.csv')
            assert cost == -step.reward, "Cost computed by heur() method and reward are different"

            done = step.terminal

    env.close()

########################################################################################################################


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
@wrap_experiment(log_dir='gaussian-vpg/single-instance-abs-cgrid')
def train_rl_algo(ctxt=None, test_split=0.25, num_epochs=1000):

    # A trainer provides a default TensorFlow session using python context
    with TFTrainer(snapshot_config=ctxt) as trainer:

        # Load data from file
        predictions = pd.read_csv('instancesPredictions.csv')
        realizations = pd.read_csv('instancesRealizations.csv')
        shift = np.load('optShift.npy')
        cGrid = np.load('gmePrices.npy')

        # Split between training and test
        if isinstance(test_split, float):
            split_index = int(len(predictions) * (1 - test_split))
            train_predictions = predictions[:split_index]
            train_realizations = realizations[:split_index]
        elif isinstance(test_split, list):
            split_index = test_split
            train_predictions = predictions.iloc[split_index]
            train_realizations = realizations.iloc[split_index]
        else:
            raise Exception("test_split must be int or float")

        # Create the environment
        env = VPPEnv(predictions=train_predictions,
                     realizations=train_realizations,
                     shift=shift,
                     cGrid=cGrid)

        # garage wrapping of a gym environment
        env = GymEnv(env, max_episode_length=1)
        timestamps = timestamps_headers(env.n)

        # A policy represented by a Gaussian distribution which is parameterized by a multilayer perceptron (MLP)
        policy = GaussianMLPPolicy(env.spec)

        # A linear value function (baseline) based on features
        baseline = LinearFeatureBaseline(env_spec=env.spec)

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
        trainer.train(n_epochs=num_epochs, batch_size=1, plot=False)

        # Test the trained model on the test istances

        print('\n\n')
        print("Testing of the trained algorithm...")

        last_obs, _ = env.reset()
        policy = algo.policy
        policy.reset()
        episode_length = 0

        while episode_length < np.inf:
            env.render(mode='ascii')
            a, agent_info = policy.get_action(last_obs)
            # Choose a deterministic action when testing
            a = agent_info['mean']
            print('Action')
            print(tabulate(np.expand_dims(a, axis=0), headers=timestamps, tablefmt='pretty'))
            step = env.step(a)
            episode_length += 1
            if step.last:
                break
            last_obs = step.observation


########################################################################################################################


def test_rl_algo():
    test_split = 0.25

    # Load data from file
    predictions = pd.read_csv('instancesPredictions.csv')
    realizations = pd.read_csv('instancesRealizations.csv')
    shift = np.load('optShift.npy')
    cGrid = np.load('gmePrices.npy')

    # Split between training and test
    split_index = int(len(predictions) * (1 - test_split))
    train_predictions = predictions[:split_index]
    train_realizations = realizations[:split_index]
    test_predictions = predictions[split_index:]
    test_realizations = realizations[split_index:]

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        data = cloudpickle.load(open('max-cgrid-upper-bound/params.pkl', 'rb'))
        algo = data['algo']
        env = VPPEnv(predictions=test_predictions,
                     realizations=test_realizations,
                     shift=shift,
                     cGrid=cGrid)

        last_obs = env.reset()
        policy = algo.policy
        policy.reset()
        episode_length = 0
        timestamps = timestamps_headers(env.n)

        while episode_length < np.inf:
            env.render(mode='ascii')
            a, agent_info = policy.get_action(last_obs)
            a = agent_info['mean']
            print('Action')
            print(tabulate(np.expand_dims(a, axis=0), headers=timestamps, tablefmt='pretty'))
            obs, reward, done, info = env.step(a)
            episode_length += 1
            if done:
                break
            last_obs = obs


########################################################################################################################

if __name__ == '__main__':
    check_env()
    # train_rl_algo(test_split=[0], num_epochs=1000)
    # check_markovian_env()