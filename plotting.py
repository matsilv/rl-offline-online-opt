# Author: Mattia Silvestri

"""
    Methods to make plots and visualize results.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import itertools
import os
from utility import instances_preprocessing, timestamps_headers


########################################################################################################################

def _read_solution(filename, iter_idx, all_solutions):
    """
    Utility function to read a solution.
    :param filename: string; where the solution is loaded from.
    :param iter_idx: int; index of the current iteration.
    :param all_solutions: numpy.array; array with all the solutions.
    :return:
    """
    solution = pd.read_csv(filename, index_col=0)

    if iter_idx == 0:
        assert all_solutions is None, "Solutions array must be empty at first iteration"
        all_solutions = np.expand_dims(solution, axis=0)
    else:
        assert all_solutions is not None, "Solutions array must be non-empty after first iteration"
        solution = np.expand_dims(solution, axis=0)
        all_solutions = np.concatenate((all_solutions, solution), axis=0)

    return all_solutions

########################################################################################################################


def _plot_mean_std_solution(dict_mean,
                            dict_std,
                            columns,
                            max_values_df,
                            min_values_df):
    """
    Plot a mosaic with all the decision variables and for all the methods.
    :param dict_mean: dictionary of pandas.Dataframe; a dictionary where the keys are the methods name and the values
                                                      are pandas.Dataframe with the mean of the decision variables
                                                      computed on the instances.
    :param dict_std: dictionary of pandas.Dataframe; a dictionary where the keys are the methods name and the values
                                                      are pandas.Dataframe with the std dev of the decision variables
                                                      computed on the instances.
    :param columns: list of string; the decision variables names.
    :param max_values_df: pandas.Dataframe; max value for each decision variable.
    :param min_values_df: pandas.Dataframe; min value for each decision variable.
    :return:
    """

    assert len(dict_mean) == len(dict_std), "Mean and std dictionaries must have the same size"
    assert dict_mean.keys() == dict_std.keys(), "Mean and std dictionaries keys must be the same"

    # Rename some variables names to be more concise
    rename_dict = dict()
    rename_dict['Diesel power consumption'] = 'Diesel'
    rename_dict['Input to storage'] = 'Storage input'
    rename_dict['Output from storage'] = 'Storage output'

    # Get methods names
    methods = dict_mean.keys()

    sns.set_style('darkgrid')

    # Create subplots and set padding between frames
    fig, axises = plt.subplots(len(columns), len(methods), sharex=True, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Get all pairs methods-columns
    mosaic = itertools.product(methods, columns)

    # Iterate over all combinations
    for i, element in enumerate(mosaic):
        method, col = element[0], element[1]

        # Get max and min value to set axis limits
        max_val = max_values_df[col]
        min_val = min_values_df[col]
        max_val = max_val + max_val * 0.2
        min_val = min_val - np.sign(min_val) * min_val * 0.2

        # Renaming
        if col in rename_dict.keys():
            renamed_col = rename_dict[col]
        else:
            renamed_col = col

        # Select the frame
        i = np.unravel_index(i, (len(methods), len(columns)))
        axis = axises[i]

        # Set methods and decision variables labels
        if i[1] == 0:
            axis.set_ylabel(method, rotation=0, fontsize=12, fontweight='bold')
            if '\n' in method:
                y_coor = .2
            else:
                y_coor = .4
            axis.yaxis.set_label_coords(-.5, y_coor)
        if i[0] == 0:
            axis.set_title(renamed_col, fontsize=12, fontweight='bold')

        # Get mean and std dev for the method and decision variable
        df_mean = dict_mean[method]
        df_std = dict_std[method]
        mean_values = df_mean[col].values
        std_values = df_std[col].values

        # Set the y limits and make the plots
        axis.set_ylim(min_val, max_val)
        axis.plot(df_mean.index, mean_values, label=col)
        axis.fill_between(df_mean.index,
                          mean_values - std_values,
                          mean_values + std_values,
                          color='lightskyblue')
        axis.set_xticks([])
        axis.set_yticks([])

    plt.show()

########################################################################################################################


def cost_over_time(indexes, durations_filepath, models_filepath):
    """
    Plot the normalized cost over time for hybrid rl/opt single step and MDP and the pure RL approach.
    :param indexes: np.array; instances indexes.
    :param durations_filepath: string; where the epochs durations are loaded from.
    :param models_filepath: string; where the progress files are loaded from.
    :return:
    """

    sns.set_style('darkgrid')

    # Load all the epochs durations
    single_step_durations = np.load(os.path.join(durations_filepath, 'single-step.npy'))
    mdp_durations = np.load(os.path.join(durations_filepath, 'mdp.npy'))
    rl_durations = np.load(os.path.join(durations_filepath, 'rl.npy'))

    # Compute the mean epoch duration for each method
    single_step_epoch_duration = np.mean(single_step_durations)
    mdp_epoch_duration = np.mean(mdp_durations)
    rl_epoch_duration = np.mean(rl_durations)

    print(f'[Epoch duration] - Single step: {single_step_epoch_duration} | MDP: {mdp_epoch_duration} | RL: {rl_epoch_duration}')

    # Iterate over all the instances indexes
    for i, idx in enumerate(indexes):

        # Load the progress file for each method
        single_step = pd.read_csv(os.path.join(models_filepath,
                                               'hybrid-rl-opt',
                                               f'single-step-env_{idx}',
                                               'progress.csv'))
        mdp = pd.read_csv(os.path.join(models_filepath,
                                       'hybrid-rl-opt',
                                       f'mdp-env_{idx}',
                                       'progress.csv'))
        rl = pd.read_csv(os.path.join(models_filepath,
                                      'pure-rl',
                                      f'mdp-env_{idx}',
                                      'progress.csv'))
        oracle = np.load(os.path.join(models_filepath,
                                      'oracle',
                                      f'{idx}_cost.npy'))
        robust_kkt = np.load(os.path.join(models_filepath,
                                          'robust-kkt',
                                          f'id-{idx}',
                                          f'{idx}_cost.npy'))
        simple_greedy_heuristic = np.load(os.path.join(models_filepath,
                                                       'simple-greedy-heuristic',
                                                       f'{idx}_cost.npy'))

        # Get the episode mean reward
        single_step = -single_step['Extras/EpisodeRewardMean']
        mdp = -mdp['Extras/EpisodeRewardMean']
        rl = -rl['Extras/EpisodeRewardMean']

        # Find the maximum reward to compute normalization
        all_reward_mean = np.concatenate((single_step, mdp, rl), axis=0)
        all_reward_mean = np.append(all_reward_mean, [oracle, robust_kkt, simple_greedy_heuristic], axis=0)
        min_reward_mean = np.min(all_reward_mean)

        # Normalize each method rewards
        single_step = single_step / min_reward_mean
        mdp = mdp / min_reward_mean
        rl = rl / min_reward_mean
        oracle = oracle / min_reward_mean
        robust_kkt = robust_kkt / min_reward_mean
        simple_greedy_heuristic = simple_greedy_heuristic / min_reward_mean

        # Create a single numpy array for each method with all the instances
        if i == 0:
            normalized_single_step = np.expand_dims(single_step, axis=0)
            normalized_mdp = np.expand_dims(mdp, axis=0)
            normalized_rl = np.expand_dims(rl, axis=0)
            normalized_robust_kkt = np.expand_dims(robust_kkt, axis=0)
            normalized_simple_greedy_heuristic = np.expand_dims(simple_greedy_heuristic, axis=0)
            normalized_oracle = np.expand_dims(oracle, axis=0)
        else:
            single_step = np.expand_dims(single_step, axis=0)
            mdp = np.expand_dims(mdp, axis=0)
            rl = np.expand_dims(rl, axis=0)
            normalized_single_step = np.concatenate((normalized_single_step, single_step), axis=0)
            normalized_mdp = np.concatenate((normalized_mdp, mdp), axis=0)
            normalized_rl = np.concatenate((normalized_rl, rl), axis=0)
            robust_kkt = np.expand_dims(robust_kkt, axis=0)
            normalized_robust_kkt = np.concatenate((normalized_robust_kkt, robust_kkt), axis=0)
            simple_greedy_heuristic = np.expand_dims(simple_greedy_heuristic, axis=0)
            normalized_simple_greedy_heuristic = np.concatenate((normalized_simple_greedy_heuristic,
                                                                 simple_greedy_heuristic),
                                                                axis=0)
            oracle = np.expand_dims(oracle, axis=0)
            normalized_oracle = np.concatenate((normalized_oracle, oracle), axis=0)

    # Compute the mean for each method on the instances axis
    normalized_single_step = np.mean(normalized_single_step, axis=0)
    normalized_mdp = np.mean(normalized_mdp, axis=0)
    normalized_rl = np.mean(normalized_rl, axis=0)
    normalized_robust_kkt = np.mean(normalized_robust_kkt, axis=0)
    normalized_simple_greedy_heuristic = np.mean(normalized_simple_greedy_heuristic, axis=0)
    normalized_oracle = np.mean(normalized_oracle, axis=0)

    # Convert epoch to durations in seconds
    single_step_time = np.arange(len(normalized_single_step)) * single_step_epoch_duration
    mdp_time = np.arange(len(normalized_mdp)) * mdp_epoch_duration
    rl_time = np.arange(len(normalized_rl)) * rl_epoch_duration

    # Find the minimum duration so that rewards array have the same durations
    min_duration = min(single_step_time[-1], mdp_time[-1], rl_time[-1])
    print(f'[Training end] - Single step: {single_step_time[-1]} | MDP: {mdp_time[-1]} | RL: {rl_time[-1]}')
    print(f'[Epochs] - Single step: {min_duration / single_step_epoch_duration} | MDP: {min_duration / mdp_epoch_duration} | RL: {min_duration / rl_epoch_duration}')
    single_step_time = single_step_time[single_step_time <= min_duration]
    normalized_single_step = normalized_single_step[:len(single_step_time)]
    mdp_time = mdp_time[mdp_time <= min_duration]
    normalized_mdp = normalized_mdp[:len(mdp_time)]
    rl_time = rl_time[rl_time <= min_duration]
    normalized_rl = normalized_rl[:len(rl_time)]

    # Since the RL method is the most dense array, we perform interpolation for the other methods
    normalized_single_step = np.interp(rl_time, single_step_time, normalized_single_step)
    normalized_mdp = np.interp(rl_time, mdp_time, normalized_mdp)

    # Plots the results
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    plt.subplots_adjust(wspace=0, hspace=0.05)

    ax1.plot(rl_time, normalized_rl, label='rl')
    ax1.axhline(normalized_oracle, label='oracle', linestyle='--', color='red')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(fontsize=12)

    ax2.axhline(normalized_robust_kkt,
                label='tuning',
                linestyle='dotted',
                color='b')
    ax2.axhline(normalized_simple_greedy_heuristic,
                label='greedy-heuristic',
                linestyle=(0, (1, 10)),
                color='g')
    ax2.axhline(normalized_oracle,
                label='oracle',
                linestyle='--',
                color='r')
    ax2.plot(rl_time,
             normalized_single_step,
             label='single-step',
             linestyle='solid',
             color='c')
    ax2.plot(rl_time,
             normalized_mdp,
             label='mdp',
             linestyle='dashdot',
             color='m')
    ax2.set_xlabel('Time (sec)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(fontsize=12)

    # plt.title('Mean Episode Reward', fontsize=12, fontweight='bold')
    plt.show()

########################################################################################################################


def plot_solutions(indexes, solutions_filepath):
    """
    Plot the decision variables trend over time for all the methods.
    :param indexes: numpy.array; the numpy array with the indexes.
    :param solutions_filepath: string; the solutions base folder.
    :return:
    """
    # Instantiate the numpy arrays with all the solutions for each method
    all_single_step_solutions = None
    all_mdp_solutions = None
    all_rl_solutions = None
    all_simple_greedy_heuristic_solutions = None
    all_oracle_solutions = None
    all_robust_kkt_solutions = None

    # Read a solution sample to get the columns and index
    sample_filename = os.path.join(solutions_filepath,
                                   'hybrid-rl-opt',
                                   f'single-step-env_{indexes[0]}',
                                   f'{indexes[0]}_solution.csv')
    sample = pd.read_csv(sample_filename, index_col=0)

    # Iterate over all the instances indexes
    for i, idx in enumerate(indexes):

        # Load the progress file for each method
        single_step_filename = os.path.join(solutions_filepath,
                                            'hybrid-rl-opt',
                                            f'single-step-env_{idx}',
                                            f'{idx}_solution.csv')
        mdp_filename = os.path.join(solutions_filepath,
                                    'hybrid-rl-opt',
                                    f'mdp-env_{idx}',
                                    f'{idx}_solution.csv')
        rl_filename = os.path.join(solutions_filepath,
                                   'pure-rl',
                                   f'mdp-env_{idx}',
                                   f'{idx}_solution.csv')
        simple_greedy_heuristic_filename = os.path.join(solutions_filepath,
                                                        'simple-greedy-heuristic',
                                                        f'{idx}_solution.csv')
        oracle_filename = os.path.join(solutions_filepath,
                                       'oracle',
                                       f'{idx}_solution.csv')
        robust_kkt_filename = os.path.join(solutions_filepath,
                                           'robust-kkt',
                                           f'id-{idx}',
                                           f'{idx}_solution.csv')

        # Check that the file exists (a solutions was found)
        if os.path.exists(single_step_filename):
            all_single_step_solutions = _read_solution(filename=single_step_filename,
                                                       iter_idx=i,
                                                       all_solutions=all_single_step_solutions)

        if os.path.exists(mdp_filename):
            all_mdp_solutions = _read_solution(filename=mdp_filename,
                                               iter_idx=i,
                                               all_solutions=all_mdp_solutions)

        if os.path.exists(rl_filename):
            all_rl_solutions = _read_solution(filename=rl_filename,
                                              iter_idx=i,
                                              all_solutions=all_rl_solutions)
        if os.path.exists(simple_greedy_heuristic_filename):
            all_simple_greedy_heuristic_solutions = _read_solution(filename=simple_greedy_heuristic_filename,
                                                                   iter_idx=i,
                                                                   all_solutions=all_simple_greedy_heuristic_solutions)
        if os.path.exists(oracle_filename):
            all_oracle_solutions = _read_solution(filename=oracle_filename,
                                                  iter_idx=i,
                                                  all_solutions=all_oracle_solutions)
        if os.path.exists(robust_kkt_filename):
            all_robust_kkt_solutions = _read_solution(filename=robust_kkt_filename,
                                                      iter_idx=i,
                                                      all_solutions=all_robust_kkt_solutions)

    # Compute mean and std dev of the decision variables
    mean_single_step_solutions = np.mean(all_single_step_solutions, axis=0)
    mean_mdp_solutions = np.mean(all_mdp_solutions, axis=0)
    mean_rl_solutions = np.mean(all_rl_solutions, axis=0)
    mean_simple_greedy_heuristic_solutions = np.mean(all_simple_greedy_heuristic_solutions, axis=0)
    mean_oracle_solutions = np.mean(all_oracle_solutions, axis=0)
    mean_robust_kkt_solutions = np.mean(all_robust_kkt_solutions, axis=0)

    std_single_step_solutions = np.std(all_single_step_solutions, axis=0)
    std_mdp_solutions = np.std(all_mdp_solutions, axis=0)
    std_rl_solutions = np.std(all_rl_solutions, axis=0)
    std_simple_greedy_heuristic_solutions = np.std(all_simple_greedy_heuristic_solutions, axis=0)
    std_oracle_solutions = np.std(all_oracle_solutions, axis=0)
    std_robust_kkt_solutions = np.std(all_robust_kkt_solutions, axis=0)

    # Create a dataframe with the mean and std dev of the decision variables
    columns = sample.columns
    index = sample.index

    df_mean_single_step = pd.DataFrame(mean_single_step_solutions,
                                       index=index,
                                       columns=columns)
    df_mean_mdp = pd.DataFrame(mean_mdp_solutions,
                               index=index,
                               columns=columns)
    df_mean_rl = pd.DataFrame(mean_rl_solutions,
                              index=index,
                              columns=columns)
    df_mean_simple_greedy_heuristic = pd.DataFrame(mean_simple_greedy_heuristic_solutions,
                                                   index=index,
                                                   columns=columns)
    df_mean_robust_kkt = pd.DataFrame(mean_robust_kkt_solutions,
                                      index=index,
                                      columns=columns)
    df_mean_oracle = pd.DataFrame(mean_oracle_solutions,
                                  index=index,
                                  columns=columns)

    df_std_single_step = pd.DataFrame(std_single_step_solutions,
                                      index=index,
                                      columns=columns)
    df_std_mdp = pd.DataFrame(std_mdp_solutions,
                              index=index,
                              columns=columns)
    df_std_rl = pd.DataFrame(std_rl_solutions,
                             index=index,
                             columns=columns)
    df_std_simple_greedy_heuristic = pd.DataFrame(std_simple_greedy_heuristic_solutions,
                                                  index=index,
                                                  columns=columns)
    df_std_robust_kkt = pd.DataFrame(std_robust_kkt_solutions,
                                     index=index,
                                     columns=columns)
    df_std_oracle = pd.DataFrame(std_oracle_solutions,
                                 index=index,
                                 columns=columns)

    # Create dictionaries with (key - decision variables mean) and (key - decision variables std dev)
    dict_mean = dict()
    dict_mean['Single step'] = df_mean_single_step
    dict_mean['MDP'] = df_mean_mdp
    dict_mean['RL'] = df_mean_rl
    dict_mean['Greedy\nheuristic'] = df_mean_simple_greedy_heuristic
    dict_mean['Tuning'] = df_mean_robust_kkt
    dict_mean['Oracle'] = df_mean_oracle

    dict_std = dict()
    dict_std['Single step'] = df_std_single_step
    dict_std['MDP'] = df_std_mdp
    dict_std['RL'] = df_std_rl
    dict_std['Greedy\nheuristic'] = df_std_simple_greedy_heuristic
    dict_std['Tuning'] = df_std_robust_kkt
    dict_std['Oracle'] = df_std_oracle

    # Create pandas.Dataframe with the maximum and minimum values for each variable and method
    methods = dict_mean.keys()
    max_value_df = pd.DataFrame(index=methods, columns=columns)
    min_value_df = max_value_df.copy()

    for meth in methods:
        max_value_df.loc[meth] = (dict_mean[meth] + dict_std[meth]).max(axis=0)
        min_value_df.loc[meth] = (dict_mean[meth] - dict_std[meth]).min(axis=0)
    max_value_df = max_value_df.max(axis=0)
    min_value_df = min_value_df.min(axis=0)

    _plot_mean_std_solution(dict_mean,
                            dict_std,
                            columns,
                            max_value_df,
                            min_value_df)

########################################################################################################################


def plot_pv_and_load_forecasts(loadpath):
    """
    Plot the photovoltaic and user laod demand forecasts.
    :param loadpath: string; where data are loaded from.
    :return:
    """

    # Read data and make pre-processing
    forecasts = pd.read_csv(loadpath, index_col=0)
    assert 'PV(kW)' in forecasts.columns and 'Load(kW)' in forecasts.columns
    forecasts = instances_preprocessing(forecasts)

    pv = forecasts['PV(kW)'].values
    load = forecasts['Load(kW)'].values

    pv = [np.asarray(pv_instance) for pv_instance in pv]
    load = [np.asarray(load_instance) for load_instance in load]

    pv = np.asarray(pv)
    load = np.asarray(load)

    assert pv.shape == load.shape

    # Compute mean and std dev
    mean_pv = np.mean(pv, axis=0)
    std_pv = np.std(pv, axis=0)
    mean_load = np.mean(load, axis=0)
    std_load = np.std(load, axis=0)

    timesteps = timestamps_headers(num_timeunits=pv.shape[1])

    # Make plots
    sns.set_style('darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(timesteps, mean_pv)
    ax1.fill_between(timesteps,
                     mean_pv - std_pv,
                     mean_pv + std_pv,
                     color='lightskyblue')
    ax1.set_xticks([])
    ax1.set_title('Photovoltaic production', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%d kW'))
    ax2.plot(timesteps, mean_load)
    ax2.fill_between(timesteps,
                     mean_load - std_load,
                     mean_load + std_load,
                     color='lightskyblue')
    ax2.set_xticks(timesteps[::8])
    ax2.set_xlabel('Timestamp (hh:mm)', fontsize=11)
    ax2.set_title('Load demand', fontsize=12, fontweight='bold')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%d kW'))
    plt.show()

########################################################################################################################


def compare_cost(filepath1,
                 filepath2,
                 name1,
                 name2,
                 baseline=None):
    """
    Compare the average reward over episodes between results saved in the specified filepath.
    :param filepath1: string; first results path.
    :param filepath2: string; second results path.
    :param name1: string; label name of the first plot.
    :param name2: string; label name of the second plot
    :param baseline: float; optionally, a baseline value is plotted.
    :return:
    """

    sns.set_style('darkgrid')

    # Read the episode mean reward
    rew1 = pd.read_csv(filepath1)['Extras/EpisodeRewardMean']
    rew2 = pd.read_csv(filepath2)['Extras/EpisodeRewardMean']
    rew1 = -rew1
    # rew1[rew1 > 10000] = np.nan

    rew2 = -rew2
    # rew2[rew2 > 10000] = np.nan

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d (k€)'))
    plt.title('Average episode reward', fontweight='bold')
    plt.xlabel('Epoch')

    rew1.name = name1
    rew2.name = name2

    rew1.plot()
    rew2.plot()

    # Optionally, you can plot a baseline value
    if baseline is not None:
        plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline mean cost')

    plt.legend()
    plt.show()