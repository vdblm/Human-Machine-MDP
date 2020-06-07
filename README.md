# Learning to Switch Between Machines and Humans

## Requirements
 To install requirements:

 ```setup
 pip install -r requirements.txt
 ```

## Code structure (Training and Evaluation)

 - `agents/`
     - `agents/hum_mac_agents.py` contains human/machine action policies.

     - `agents/switching_agents.py` contains implementation of Algorithm 1, Algorithm 2, UCRL2, and the Greedy algorithm in the paper.

 - `environments/` contains the code to produce all the environment types (episodic MDPs) used in the paper (i.e., Env-1, Env-2, and Env-3).

 - `experiments/` contains all the known/unknown human experiments. For example,
     - `SensorBasedSwitchingExperiment.run_unknown_human_exp` trains and evaluates Algorithm 2, UCRL2, and the Greedy algorithm in an episodic setting.


 - `plot/` contains the code to plot the figures in the paper.

## Results

See the notebook `plots.ipynb` to reproduce the results in the paper:

```notebook
jupyter notebook plots.ipynb
```
