# Hybrid Offline/Online Optimization for Energy Management via Reinforcement Learning

 
This repository contains the code to reproduce the experiments of the paper **Hybrid Offline/Online Optimization for Energy
Management via Reinforcement Learning** under review at CPAIOR-2022 conference.

## Script description
* `online_heuristic.py`: utility functions to evaluate the solutions and invoke the simple
                         greedy heuristic.
* `plotting.py`: utility functions to make plots shown in the paper.
* `tests.py`: this script contains the functions to train and test the different methods.
* `utility.py`: simple utility functions, like pre-processing.
* `vpp_envs.py`: gym environments for the VPP.

## Data folder description
* `Dataset10k.csv`: photovoltaic production and user load demand forecasts with 15 minutes
                    resolution. Each pair of forecast is called an instance. There are 
                    10,000 instances in the dataset which is obtained from 
                    [https://www.enwl.co.uk/lvns](https://www.enwl.co.uk/lvns).
* `gmePrices.npy`:  demand electricity hourly obtained based on data from the Italian national energy 
                    market management corporation (GME).
* `optShift.npy`: optimal day-ahead load demand shifts.

## Run the experiments
The experiments can be run launching the `tests.py` script. The arguments are the following:
* `logdir`: logging and model directory.
* `--mode`: training or test.
* `--method`: you can choose among the following methods:
  * `hybrid-single-step`: this is referred to as `single-step` in the paper;
  * `hybrid-mdp`: this is referred to as `mdp` in the paper;
  * `rl-single-step`: end-to-end RL approach which directly provides the decision variables for 
                      all the stages;
  * `rl-mdp`: this is referred to as `rl` in the paper.
* `--epochs`: the number of episodes; it is valid for both training and test.
* `--batch-size`: the batch size.  

The training routine is performed by the `train_rl_algo` method whereas the test is performed by
the `test_rl_algo` method. Please, be sure to have a `data` with the following files:
* `Dataset10k.csv`
* `gmePrices.npy`
* `optShift.npy`

which have been described in the previous section.

`test_rl_algo` method saves the solution or virtual costs (depending on the employed algorithm)
in the `logdir` folder with name respectively `solution.npy` and `cvirt.npy` as `numpy.array`.

To evaluate and visualize the solutions/virtual costs, run the `online_heuristic.py` script:
which has the following arguments:

* `instance_filepath`: user demand and RES productions forecasting filepath.
* `shifts_filepath`: optimal shifts filepath.
* `prices_filepath`: Electricity Market prices filepath.
* `solution_filepath`: Virtual costs or solution filepath.
* `mode`: `rl` or `hybrid`; If `rl` is selected then you have to directly provide a solution;
          if `hybrid` is provided then you have to provide the virtual costs.
* `instance_id`: instance index in the instances file.
* `savepath`: where the solution and its cost are saved to.
* `--display`: display the solution and cost.

## Issue
Currently, there is an issue with the `garage` library. If the `test.py` script in launched in the 
the same directory of the git repository then the following execption is raised
`TypeError: CreateProcess() argument 8 must be str or None, not bytes`.
Please, run the code in separate directory from the git repository. Sorry for the inconvenience.
The experiments were run on Windows 10 OS.
            

