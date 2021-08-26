# rl-offline-online-opt
 
This repository contains the code to reproduce the experiments concerning 
the adoption of an RL algorithm to set some hyperparameters of an heuristic 
function to solve stochastic optimization problems.

## Script description
* `OnlineHeuristic.py`: implementation of the original objective function.
* `OnlineHeuristic2.py`: implementation of the objective function with more
complex storage constraints.
* `rl_utils.py`: implementation of single step and MDP environment.
* `runScript.py`: run the online heuristic.
* `tests.py`: main functions to debug environment implementation and train 
the RL agent.

## Experiment description
* The RL agent is trained on a single instance. 
Untill now, we have performed experiments on the instance of index 0 of 
the file `instancesPredictions.csv`. A graphical representation of the 
instance can be found at the path `results/precition-0.txt`. The RL agent 
achieves a mean cost on 100 episoded of 20.93. 
* To train the agent, uncomment the `train_rl_algo` method. The folder 
where the experiment files are saved can be set with `log_dir` method 
of the `experiment` decorator.
* To test the agent, uncomment the `test_rl_algo` method and set the 
proper `log_dir`.

