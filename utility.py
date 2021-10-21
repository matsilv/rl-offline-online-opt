# Author: Mattia Silvestri

from garage.experiment.experiment import ExperimentTemplate

########################################################################################################################


def my_wrap_experiment(function,
                       logging_dir,
                       *,
                       prefix='experiment',
                       name=None,
                       snapshot_mode='last',
                       snapshot_gap=1,
                       archive_launch_repo=True,
                       name_parameters=None,
                       use_existing_dir=False,
                       x_axis='TotalEnvSteps'):
    return ExperimentTemplate(function=function,
                              log_dir=logging_dir,
                              prefix=prefix,
                              name=name,
                              snapshot_mode=snapshot_mode,
                              snapshot_gap=snapshot_gap,
                              archive_launch_repo=archive_launch_repo,
                              name_parameters=name_parameters,
                              use_existing_dir=use_existing_dir,
                              x_axis=x_axis)
